# MetricX inference service

A standalone, Dockerized HTTP service that keeps a **MetricX** model warm on a
dedicated GPU and scores translations with vLLM-style dynamic batching.
Imports nothing from `DataCurator`; replaces the slow per-batch subprocess in
`metricx/score_metricx.py` (which reloads the 3.7B model every call and runs
unbatched). Two variants (auto-detected from the model name, or forced via
`METRICX_VARIANT`):

- `metricx-23-qe` (`google/metricx-23-qe-xl-v2p0`, default) — reference-free:
  scores `hypothesis` against `source`.
- `metricx-24-hybrid` (`google/metricx-24-hybrid-xl-v2p6`) — additionally takes
  an optional per-pair `reference` (ground truth); pairs without one are scored
  QE-style.

## API

```
POST /score   {"pairs": [{"source": "...", "hypothesis": "...", "reference"?: "..."}, ...]}
           -> {"scores": [float, ...]}          # 0–25, lower is better, order-aligned
GET  /health  -> {"status", "model_loaded", "accepts_reference"}   # 200 once loaded
GET  /metrics -> Prometheus (batch size, queue depth, latency, in-flight)
GET  /docs    -> Swagger UI (OpenAPI spec at /openapi.json)
```

Sending `reference` to a QE deployment is a 400 (check `/health`). Set an
optional `METRICX_TOKEN` to require `Authorization: Bearer <token>`.

## Run

```bash
cp .env.example .env          # set METRICX_PUBLIC_HOST and METRICX_TOKEN
docker compose up --build     # first start downloads ~15GB into the hf-cache volume
curl localhost:8000/health
curl -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"pairs":[{"source":"The cat sat.","hypothesis":"Kot usiadł."}]}'
```

## How it works

- **`metricx_model.py`** — a committed copy of upstream `MT5ForRegression`
  (Apache-2.0; identical for MetricX-23 and -24; reads the score from the
  `<extra_id_10>` logit clamped to `[0,25]`) plus `MetricXScorer`, which
  reproduces `predict.py` exactly (input templates in `variants.py`, EOS drop)
  but **pads a whole batch** into one forward pass — the batching upstream
  can't do.
- **`batcher.py`** — one async queue feeding one inference worker; concurrent
  requests coalesce into GPU-efficient batches (`MAX_BATCH_SIZE` /
  `MAX_BATCH_WAIT_MS`), and the blocking forward pass runs off the event loop so
  the next batch forms while the GPU is busy.
- **`server.py`** — Starlette + uvicorn; loads the model on a background thread
  so the HTTP surface is up (returning 503) while weights load.

## Config

See `.env.example`. Key knobs: `METRICX_MODEL` + `METRICX_VARIANT`
(auto-detected from the model name; sets the default max input length — 1024
for 23, 1536 for 24), `METRICX_DTYPE` (`float32` reproduces the offline eval
numbers; `float16`/`bfloat16` halve VRAM and latency but shift scores slightly
— recalibrate), `CUDA_VISIBLE_DEVICES`, `MAX_BATCH_SIZE`, `MAX_BATCH_WAIT_MS`.

## Tests

`tests/test_server.py` covers the HTTP + batching logic with a stub scorer (no
torch needed). On a GPU box, validate faithfulness by comparing a few
`POST /score` results against `metricx/score_metricx.py` on the same rows — they
must match within floating-point tolerance.
