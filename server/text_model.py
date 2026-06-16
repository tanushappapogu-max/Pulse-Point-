"""
TeaTextCNN — 1D Convolutional Neural Network for Tea Text Classification

Architecture: Kim (2014) "Convolutional Neural Networks for Sentence Classification"
  https://arxiv.org/abs/1408.5882

Multi-task variant: simultaneous tea-type classification (single-label) and
flavor-profile detection (multi-label), plus quality-tier prediction.

Input:  Raw text string (tea description, product label, spoken query)
Output:
  - tea_type: one of 8 types (green/black/white/oolong/pu_erh/herbal/yellow/dark)
  - flavors:  multi-label set of 10 flavor notes
  - quality:  one of 4 quality tiers (ceremonial/premium/standard/culinary)
  - embedding: 256-dim sentence vector (for downstream similarity tasks)
"""

import os
import re
import json
import math
import pickle
import string
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from tea_dataset import (
    TEA_TYPES, FLAVOR_LABELS, QUALITY_TIERS,
    build_augmented_dataset,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = Path(__file__).parent
VOCAB_PATH  = _DIR / "tea_vocab.pkl"
MODEL_PATH  = _DIR / "tea_textcnn.pt"


# ═══════════════════════════════════════════════════════════════════════════
# §1  VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Character-free word vocabulary with special tokens."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = [self.PAD, self.UNK]
        self._freq: dict[str, int] = {}

    # -- construction -------------------------------------------------------

    def fit(self, texts):
        for text in texts:
            for tok in self._tokenize(text):
                self._freq[tok] = self._freq.get(tok, 0) + 1
        for word, freq in sorted(self._freq.items()):
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
        return self

    # -- encoding -----------------------------------------------------------

    def encode(self, text, max_len=64):
        tokens = self._tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts, max_len=64):
        return [self.encode(t, max_len) for t in texts]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return text.split()

    def __len__(self):
        return len(self.idx2word)

    # -- persistence --------------------------------------------------------

    def save(self, path=VOCAB_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path=VOCAB_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# §2  TEXTCNN ARCHITECTURE  (Kim, 2014)
#
#   Embedding → parallel Conv1d filters at multiple n-gram widths →
#   max-over-time pooling → concat → Dropout → multi-task heads
#
#   Filter widths [2,3,4,5] capture bigram through 5-gram patterns.
#   Each width has `num_filters` feature maps.
#   Output feature vector: len(filter_sizes) × num_filters = 256 dims.
# ═══════════════════════════════════════════════════════════════════════════

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        filter_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.4,
        num_tea_types: int = len(TEA_TYPES),
        num_flavors: int = len(FLAVOR_LABELS),
        num_quality: int = len(QUALITY_TIERS),
        max_len: int = 64,
    ):
        super().__init__()
        self.max_len = max_len
        self.filter_sizes = filter_sizes
        feature_dim = num_filters * len(filter_sizes)

        # ── Embedding ──────────────────────────────────────────────────────
        # Index 0 = <pad>; padding_idx keeps its embedding frozen at zero
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        # ── Parallel Conv1d towers ─────────────────────────────────────────
        # Input to Conv1d: [B, embed_dim, seq_len]  (channels-first)
        # Each tower: Conv1d → BatchNorm1d → ReLU → max-over-time pool → [B, num_filters]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,   # same-ish padding keeps seq_len stable
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
            )
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # ── Multi-task classification heads ────────────────────────────────
        self.tea_type_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_tea_types),
        )

        self.flavor_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_flavors),  # BCEWithLogitsLoss (multi-label)
        )

        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_quality),
        )

        # ── Projection to shared embedding space ───────────────────────────
        self.embed_projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _conv_and_pool(self, x):
        """x: [B, seq_len, embed_dim]  →  feature: [B, num_filters * len(filter_sizes)]"""
        # Conv1d expects [B, C, L] — permute embed to channel dim
        x = x.permute(0, 2, 1)  # [B, embed_dim, seq_len]

        pooled = []
        for conv in self.convs:
            out = conv(x)            # [B, num_filters, seq_len']
            out = out.max(dim=2).values  # max-over-time: [B, num_filters]
            pooled.append(out)

        return torch.cat(pooled, dim=1)  # [B, feature_dim]

    def forward(self, input_ids):
        """
        Args:
            input_ids: LongTensor [B, seq_len]
        Returns:
            tea_logits:    [B, num_tea_types]
            flavor_logits: [B, num_flavors]     (raw logits for BCE)
            quality_logits:[B, num_quality]
            sentence_emb:  [B, 256]             (L2-normalised)
        """
        emb = self.embedding(input_ids)   # [B, seq_len, embed_dim]
        features = self._conv_and_pool(emb)
        features = self.dropout(features)

        tea_logits     = self.tea_type_head(features)
        flavor_logits  = self.flavor_head(features)
        quality_logits = self.quality_head(features)
        sentence_emb   = F.normalize(self.embed_projection(features), dim=-1)

        return tea_logits, flavor_logits, quality_logits, sentence_emb


# ═══════════════════════════════════════════════════════════════════════════
# §3  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TeaTextDataset(torch.utils.data.Dataset):
    def __init__(self, examples, vocab, max_len=64):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = self.vocab.encode(ex["text"], self.max_len)
        return {
            "input_ids":    torch.tensor(ids, dtype=torch.long),
            "tea_type":     torch.tensor(ex["tea_type"], dtype=torch.long),
            "flavors":      torch.tensor(ex["flavors"],  dtype=torch.float),
            "quality":      torch.tensor(ex["quality"],  dtype=torch.long),
        }


def train(
    epochs=120,
    batch_size=16,
    lr=3e-3,
    weight_decay=1e-4,
    embed_dim=64,
    num_filters=64,
    filter_sizes=(2, 3, 4, 5),
    max_len=64,
    seed=42,
):
    """Train TextCNN on the tea dataset and save artefacts to disk."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    examples = build_augmented_dataset(seed=seed)

    # -- Build vocabulary ---------------------------------------------------
    vocab = Vocabulary(min_freq=1)
    vocab.fit([ex["text"] for ex in examples])
    vocab.save()
    print(f"Vocabulary size: {len(vocab)}")

    # -- Dataset / DataLoader -----------------------------------------------
    dataset = TeaTextDataset(examples, vocab, max_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # -- Model --------------------------------------------------------------
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        max_len=max_len,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"TextCNN | {total:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            tea_logits, flavor_logits, quality_logits, _ = model(batch["input_ids"])

            loss_tea     = ce_loss(tea_logits,     batch["tea_type"])
            loss_flavor  = bce_loss(flavor_logits, batch["flavors"])
            loss_quality = ce_loss(quality_logits, batch["quality"])

            # Weight: type classification is primary task
            loss = 1.2 * loss_tea + 0.8 * loss_flavor + 0.6 * loss_quality
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch["input_ids"])

        scheduler.step()
        avg = epoch_loss / len(dataset)

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "num_filters": num_filters,
                "filter_sizes": list(filter_sizes),
                "max_len": max_len,
                "best_loss": best_loss,
                "epoch": epoch,
            }, MODEL_PATH)

        if epoch % 20 == 0 or epoch == epochs:
            acc = _quick_accuracy(model, dataset)
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg:.4f}  tea_acc={acc:.1%}  (best={best_loss:.4f})")

    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved vocab → {VOCAB_PATH}")
    return model, vocab


def _quick_accuracy(model, dataset):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ex in dataset:
            ids = ex["input_ids"].unsqueeze(0)
            tea_logits, *_ = model(ids)
            pred = tea_logits.argmax(dim=1).item()
            if pred == ex["tea_type"].item():
                correct += 1
            total += 1
    model.train()
    return correct / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §4  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_text_model: TextCNN | None = None
_vocab: Vocabulary | None = None


def _load_artefacts():
    global _text_model, _vocab

    if _text_model is not None:
        return _text_model, _vocab

    # Auto-train if artefacts are missing
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        print("Tea TextCNN artefacts not found — running initial training…")
        _text_model, _vocab = train()
        return _text_model, _vocab

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    _vocab = Vocabulary.load(VOCAB_PATH)

    _text_model = TextCNN(
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        num_filters=ckpt["num_filters"],
        filter_sizes=tuple(ckpt["filter_sizes"]),
        max_len=ckpt["max_len"],
    )
    _text_model.load_state_dict(ckpt["model_state"])
    _text_model.eval()

    total = sum(p.numel() for p in _text_model.parameters())
    print(f"TeaTextCNN loaded | {total:,} params | best_loss={ckpt['best_loss']:.4f} (epoch {ckpt['epoch']})")

    return _text_model, _vocab


def classify_text(text: str, top_k: int = 3) -> dict:
    """
    Classify a tea-related text string.

    Returns:
        {
          "tea_type":    {"label": "green", "confidence": 0.92},
          "flavors":     [{"label": "grassy", "confidence": 0.88}, ...],
          "quality":     {"label": "premium", "confidence": 0.71},
          "alternatives": [{"label": ..., "confidence": ...}, ...],
          "embedding":   [float, ...]   # 256-dim sentence vector
        }
    """
    model, vocab = _load_artefacts()

    ids = torch.tensor([vocab.encode(text, model.max_len)], dtype=torch.long)

    with torch.no_grad():
        tea_logits, flavor_logits, quality_logits, sentence_emb = model(ids)

        # -- Tea type (softmax) -------------------------------------------
        tea_probs = F.softmax(tea_logits[0], dim=0)
        best_type_idx = tea_probs.argmax().item()
        best_type_conf = tea_probs[best_type_idx].item()

        top_probs, top_idxs = tea_probs.topk(min(top_k, len(TEA_TYPES)))
        alternatives = [
            {"label": TEA_TYPES[i.item()], "confidence": round(p.item(), 4)}
            for p, i in zip(top_probs, top_idxs)
        ]

        # -- Flavors (sigmoid, multi-label) --------------------------------
        flavor_probs = torch.sigmoid(flavor_logits[0])
        FLAVOR_THRESHOLD = 0.35
        detected_flavors = [
            {"label": FLAVOR_LABELS[i], "confidence": round(flavor_probs[i].item(), 4)}
            for i in range(len(FLAVOR_LABELS))
            if flavor_probs[i].item() >= FLAVOR_THRESHOLD
        ]
        detected_flavors.sort(key=lambda x: -x["confidence"])

        # -- Quality tier --------------------------------------------------
        quality_probs = F.softmax(quality_logits[0], dim=0)
        best_q_idx = quality_probs.argmax().item()
        best_q_conf = quality_probs[best_q_idx].item()

    return {
        "tea_type": {
            "label": TEA_TYPES[best_type_idx],
            "confidence": round(best_type_conf, 4),
        },
        "flavors": detected_flavors if detected_flavors else [
            {"label": FLAVOR_LABELS[flavor_probs.argmax().item()], "confidence": round(flavor_probs.max().item(), 4)}
        ],
        "quality": {
            "label": QUALITY_TIERS[best_q_idx],
            "confidence": round(best_q_conf, 4),
        },
        "alternatives": alternatives,
        "embedding": sentence_emb[0].tolist(),
    }


def retrain():
    """Re-train from scratch and reload into global singleton."""
    global _text_model, _vocab
    _text_model, _vocab = train()
    return {"status": "ok", "vocab_size": len(_vocab)}
