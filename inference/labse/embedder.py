"""LaBSE embedder + batched encode for the inference service.

The GPU-bound half of margin-based bilingual filtering (see
``wiki/MarginBasedSimilarity.md``): LaBSE maps a sentence in any language into
one shared 768-dim space, and margin scoring is done downstream on the vectors.
Embedding is per-sentence independent, so — exactly like the MetricX service —
concurrent requests coalesce into one padded forward pass via the shared
``DynamicBatcher``; this module just owns the model.

``sentence-transformers`` is used deliberately rather than raw ``transformers``:
LaBSE's sentence vector is the CLS token pushed through a trained dense pooler
and then L2-normalized, and the ``SentenceTransformer`` wrapper applies that
pipeline correctly out of the box (hand-rolling the pooling is the classic
footgun). Vectors are returned **unit-norm** so a downstream inner product is
cosine similarity — the only similarity margin scoring ever needs.

Mirrors ``inference/metricx/metricx_model.py``: a ``load`` classmethod, a
``load_embedder_from_env`` used as the server's ``model_loader`` (run on a
background thread so the HTTP surface is up while weights download), and the
same ``METRICX_DTYPE``-style knobs. Imports torch/sentence-transformers, so —
like ``metricx_model`` — it is never imported by the unit tests.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class LabseEmbedder:
    """Loads LaBSE once and encodes a batch of texts into unit-norm vectors."""

    def __init__(
        self,
        model: SentenceTransformer,
        *,
        dim: int,
    ) -> None:
        self.model = model
        self.dim = dim

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        device: Optional[str] = None,
        dtype: str = "float32",
        max_seq_length: Optional[int] = None,
    ) -> "LabseEmbedder":
        torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = _DTYPES.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"unknown LABSE_DTYPE {dtype!r}; use one of {sorted(_DTYPES)}")

        logger.info("loading LaBSE {} on {} ({}, downloads on first run)...", model_name, torch_device, dtype)
        model = SentenceTransformer(model_name, device=torch_device)
        if torch_dtype is not torch.float32:
            # fp16/bf16 halve VRAM + latency on GPU; encode() runs in whatever
            # dtype the weights are in. (Keep float32 on CPU — half is slow there.)
            model = model.to(torch_dtype)
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        model.eval()
        dim = model.get_sentence_embedding_dimension()
        logger.info("LaBSE ready: dim={}, max_seq_length={}", dim, model.max_seq_length)
        return cls(model, dim=dim)

    @torch.inference_mode()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode ``texts`` into order-aligned, L2-normalized 768-dim vectors."""
        if not texts:
            return []
        # The DynamicBatcher already bounds the batch, so encode it in one pass
        # (batch_size >= len(texts)); normalize so dot product == cosine.
        vectors = self.model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


def load_embedder_from_env() -> LabseEmbedder:
    """Build an embedder from ``LABSE_*`` environment variables.

    Used as the server's ``model_loader``; runs on a background thread at
    startup so the HTTP surface can come up while the weights load.
    """
    max_len = os.environ.get("LABSE_MAX_SEQ_LENGTH")
    return LabseEmbedder.load(
        os.environ.get("LABSE_MODEL", "sentence-transformers/LaBSE"),
        device=os.environ.get("LABSE_DEVICE") or None,
        dtype=os.environ.get("LABSE_DTYPE", "float32"),
        max_seq_length=int(max_len) if max_len else None,
    )
