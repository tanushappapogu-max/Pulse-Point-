"""
LocateAnything-3B inference wrapper for Pulse Point.

nvidia/LocateAnything-3B is a Qwen2.5-VL-based visual grounding model.
Given an image + natural-language query it returns bounding boxes for
every matched instance in [0, 1000]-normalised coordinates.

Raw model output (JSON embedded in generated text):
    [{"bbox_2d": [x1, y1, x2, y2], "label": "cup"}]

Coordinates are integers in [0, 1000] mapped to image width/height.
"""

import io
import json
import re
import threading
import time

import torch
from PIL import Image

MODEL_ID = "nvidia/LocateAnything-3B"

_model          = None
_processor      = None
_device         = "cpu"
_load_lock      = threading.Lock()
_infer_lock     = threading.Lock()
_load_attempted = False
_load_ok        = False


def _load():
    global _model, _processor, _device, _load_ok

    from transformers import AutoProcessor, AutoModelForCausalLM

    print(f"[LocateAnything] Loading {MODEL_ID} …")
    t0 = time.time()

    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype   = torch.bfloat16 if _device == "cuda" else torch.float32

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if _device == "cuda" else None,
    )
    if _device == "cpu":
        _model = _model.to("cpu")

    _model.eval()
    _load_ok = True
    print(f"[LocateAnything] Ready in {time.time() - t0:.1f}s on {_device}")


def _ensure_loaded():
    global _load_attempted
    if _load_attempted:
        return
    with _load_lock:
        if _load_attempted:
            return
        _load_attempted = True
        try:
            _load()
        except Exception as exc:
            print(f"[LocateAnything] Load failed: {exc}")


def _parse_boxes(text: str) -> list[dict]:
    """
    Extract bbox dicts from model output.
    Handles:
      • JSON array  [{"bbox_2d": [...], "label": "..."}]
      • <box>x1,y1,x2,y2</box> tags (fallback)
    """
    # Strip special tokens that Qwen family emits
    text = re.sub(r'<\|[^|]+\|>', '', text).strip()

    # Try outermost JSON array
    bracket = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket:
        try:
            items = json.loads(bracket.group())
            if isinstance(items, list) and items and "bbox_2d" in items[0]:
                return items
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Try single JSON object
    brace = re.search(r'\{.*\}', text, re.DOTALL)
    if brace:
        try:
            obj = json.loads(brace.group())
            if "bbox_2d" in obj:
                return [obj]
        except (json.JSONDecodeError, KeyError):
            pass

    # <box>x1,y1,x2,y2</box> fallback
    box_tag = re.search(
        r'<box>\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*</box>', text
    )
    if box_tag:
        coords = [int(box_tag.group(i)) for i in range(1, 5)]
        return [{"bbox_2d": coords, "label": "object"}]

    return []


def _coords_to_bbox(coords: list[int]) -> dict:
    """
    Convert [x1, y1, x2, y2] in [0, 1000] to normalised {x, y, width, height}.
    If the values look swapped (e.g. negative width), tries [y1, x1, y2, x2].
    """
    if len(coords) != 4:
        return None

    x1, y1, x2, y2 = coords

    # Sanity check — if width is nonsensical, try the other ordering
    if x2 <= x1 or y2 <= y1:
        y1, x1, y2, x2 = coords

    x = x1 / 1000
    y = y1 / 1000
    w = (x2 - x1) / 1000
    h = (y2 - y1) / 1000

    return {
        "x":      round(max(0.0, min(x, 1.0)), 4),
        "y":      round(max(0.0, min(y, 1.0)), 4),
        "width":  round(max(0.01, min(w, 1.0)), 4),
        "height": round(max(0.01, min(h, 1.0)), 4),
    }


def locate_object(image_bytes: bytes, target: str) -> dict | None:
    """
    Run LocateAnything-3B to locate `target` in the given image.

    Returns a Pulse Point detection dict or None if nothing found / unavailable.
    """
    _ensure_loaded()
    if not _load_ok:
        return None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = f"Locate the {target}."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    try:
        text_input = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # qwen_vl_utils ships alongside the model weights; graceful fallback
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except ImportError:
            image_inputs = [image]

        inputs = _processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(_device)

        with _infer_lock, torch.inference_mode():
            output_ids = _model.generate(**inputs, max_new_tokens=256)

        generated = _processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=False,
        )[0]

        items = _parse_boxes(generated)
        if not items:
            return None

        det  = items[0]
        bbox = _coords_to_bbox(det.get("bbox_2d", []))
        if bbox is None:
            return None

        # Collect alternatives from remaining detections
        alternatives = []
        for alt in items[1:5]:
            alt_bbox = _coords_to_bbox(alt.get("bbox_2d", []))
            if alt_bbox:
                alternatives.append({
                    "name":       alt.get("label", target),
                    "confidence": 0.75,
                })

        return {
            "detected":    True,
            "name":        det.get("label", target),
            "confidence":  0.90,
            "boundingBox": bbox,
            "alternatives": alternatives,
            "model":       "LocateAnything-3B",
        }

    except Exception as exc:
        print(f"[LocateAnything] Inference error: {exc}")
        return None


def is_available() -> bool:
    return _load_ok
