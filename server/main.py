from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from locate_model import locate_object, is_available as locate_available
from model import predict, INDOOR_OBJECT_ONTOLOGY as INDOOR_OBJECTS
from text_model import classify_text, retrain
from tea_dataset import TEA_TYPES, FLAVOR_LABELS, QUALITY_TIERS

app = FastAPI(title="Pulse Point Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "locate_anything": locate_available(),
        "fallback_objects": len(INDOOR_OBJECTS),
    }


@app.get("/objects")
async def list_objects():
    return {"objects": sorted(INDOOR_OBJECTS.keys())}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    target: str = Form(default=""),
):
    start = time.time()
    image_bytes = await image.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Image too large. Max 10MB."},
        )

    result = None

    # ── Primary: LocateAnything-3B (open-vocabulary, any target) ──
    if target:
        result = locate_object(image_bytes, target)

    # ── Fallback: PulsePointNet (indoor-object ontology, ~45 classes) ──
    if result is None:
        result = predict(image_bytes, target_name=target if target else None)

    result["latency_ms"] = round((time.time() - start) * 1000)
    return result


# ── Text / Tea CNN endpoints ─────────────────────────────────────────────────

@app.post("/classify-text")
async def classify_text_endpoint(
    text: str = Body(..., embed=True, description="Tea description or spoken query"),
    top_k: int = Body(3, embed=True, description="Number of alternative tea types to return"),
):
    if not text or not text.strip():
        return JSONResponse(status_code=422, content={"error": "text must be non-empty"})
    if len(text) > 512:
        return JSONResponse(status_code=422, content={"error": "text exceeds 512 characters"})

    start = time.time()
    result = classify_text(text.strip(), top_k=top_k)
    result["latency_ms"] = round((time.time() - start) * 1000)
    result["input"] = text.strip()
    return result


@app.get("/tea-schema")
async def tea_schema():
    return {
        "tea_types":     TEA_TYPES,
        "flavor_labels": FLAVOR_LABELS,
        "quality_tiers": QUALITY_TIERS,
    }


@app.post("/retrain-text")
async def retrain_text():
    start = time.time()
    info = retrain()
    info["duration_ms"] = round((time.time() - start) * 1000)
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
