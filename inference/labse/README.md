# LaBSE embedding service

A standalone, Dockerized HTTP service that keeps **LaBSE** warm on a GPU and
returns L2-normalized multilingual sentence embeddings with vLLM-style dynamic
batching. Imports nothing from `DataCurator`. It is the GPU-bound half of
margin-based bilingual pair filtering (see `wiki/MarginBasedSimilarity.md`): the
service only embeds; the k-NN + margin math runs downstream on the vectors
(`helpers/margin_score_jsonl.py`).

Structurally a sibling of `inference/metricx/` — same batcher, same server
skeleton — because embedding, like MetricX scoring, is per-item independent and
so coalesces into GPU-efficient batches the same way.

## API

```
POST /embed   {"texts": ["...", ...]}  -> {"embeddings": [[...768...], ...], "dim": 768}
              # order-aligned, unit-norm (‖v‖≈1) so a dot product is cosine
GET  /health  -> {"status", "model_loaded", "dim", "last_error"}   # 200 once loaded
GET  /metrics -> Prometheus (batch size, queue depth, latency, in-flight)
GET  /docs    -> Swagger UI (OpenAPI spec at /openapi.json)
```

Set an optional `LABSE_TOKEN` to require `Authorization: Bearer <token>`.

## Run

```bash
cp .env.example .env          # set LABSE_PUBLIC_HOST and LABSE_TOKEN
docker compose up --build     # first start downloads ~1.8GB into the hf-cache volume
curl localhost:8024/health
curl -X POST localhost:8024/embed -H 'content-type: application/json' \
  -d '{"texts":["The cat sat.","Kot usiadł."]}'
```

## How it works

- **`embedder.py`** — `LabseEmbedder` loads `sentence-transformers/LaBSE` once and
  encodes a whole batch in one forward pass, returning **unit-norm** vectors
  (`normalize_embeddings=True`) so downstream inner products are cosine
  similarities. `sentence-transformers` (not raw `transformers`) applies LaBSE's
  dense-pooler + L2 step correctly.
- **`batcher.py`** — one async queue feeding one inference worker; concurrent
  requests coalesce into GPU-efficient batches (`MAX_BATCH_SIZE` /
  `MAX_BATCH_WAIT_MS`), and the blocking forward pass runs off the event loop so
  the next batch forms while the GPU is busy. Copied from
  `inference/metricx/batcher.py`; only the per-item value differs (a vector, not
  a score).
- **`server.py`** — Starlette + uvicorn; loads the model on a background thread so
  the HTTP surface is up (returning 503) while weights load.

## Config

See `.env.example`. Key knobs: `LABSE_MODEL`, `LABSE_MAX_SEQ_LENGTH` (truncation;
defaults to the model's own max), `LABSE_DTYPE` (`float32` default; `float16` /
`bfloat16` halve VRAM + latency on GPU), `LABSE_DEVICE`, `CUDA_VISIBLE_DEVICES`,
`MAX_BATCH_SIZE`, `MAX_BATCH_WAIT_MS`.

## Tests

`tests/test_server.py` covers the HTTP + batching logic with a stub embedder (no
torch needed). Against a real running instance, sanity-check the vectors: `dim`
is 768, each vector is unit norm (`‖v‖ ≈ 1`), identical text scores cos ≈ 1, and
an unrelated sentence scores low.
