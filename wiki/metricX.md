# Evaluating translations with MetricX-QE

This page describes how to score the Polish translations the pipeline
produces and compare them, system-by-system, against simpler baselines.
The mined trivia data has **no gold Polish references**, so quality is
measured reference-free with **MetricX-23-QE**
(`google/metricx-23-qe-xl-v2p0`), which scores `(source, hypothesis)`
pairs on a 0–25 scale where **lower is better**.

The design is deliberately manual, generic and standalone: the whole
evaluation lives in a top-level `metricx/` directory with its own venv
and imports nothing from `DataCurator`. A single scorer takes an input
(S3 prefix or local JSONL) plus the field to treat as the hypothesis,
and emits **one report per run**. You run it once per system
and combine the reports at the end. Because the pipeline's finalize
records already carry the source, the Google-Translate draft and the
final translation in the same objects, two of the three systems are
scored from the same S3 prefix — only `--hyp-field` changes.

Three systems are compared:

| System | Source of hypothesis | Field |
|--------|----------------------|-------|
| Raw Google Translate | finalize records in S3 | `text_gtranslate` |
| Simple Qwen prompt | local baseline experiment | `translation_result.translation` |
| Full gated pipeline | finalize records in S3 | `final_translation` |

All three read the same English source, `text`.

## The simple-Qwen baseline experiment

A one-line prompt against the **same** model the pipeline uses
(`qwen35_stage1_no_thinking` = `Qwen/Qwen3.6-35B-A3B-FP8`, thinking off),
so the comparison isolates "gated pipeline" from "plain prompt". It is a
single unbatched LLM stage and runs standalone against vLLM — no fleet
needed (see [Distributed](Distributed) for the batched pipeline it is
compared against).

Three additions:

1. A one-field response schema in `DataCurator/pipeline/schemas.py`:

   ```python
   class SimpleTranslation(BaseModel):
       """Single-field schema for a dead-simple translate stage."""
       translation: str = Field(description="The full Polish translation of the source text.")
   ```

2. The prompt `translate-simple.j2` in the experiment's `prompt_dir` (its only variable is `text`,
   which every input row has):

   ```jinja
   Please translate the following text into Polish:

   {{ text }}
   ```

3. The experiment `config/experiment/translate_queries_trivia_simple.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - /model/qwen35_stage1_no_thinking@_models.qwen35_stage1_no_thinking
     - /stages/llm@pipeline.translate
     - _self_

   experiment_name: translate_queries_trivia_simple

   job:
     type: data-pipeline
     links: []

   dataset:
     input: data/trivia_mined_queries.jsonl
     output_dir: dataset/${experiment_name}
     limit: null            # set e.g. 5 for a smoke run

   order:
     - translate

   pipeline:
     translate:
       enabled: true
       stage:
         name: translate
         model: ${_models.qwen35_stage1_no_thinking}
         response_model: DataCurator.pipeline.schemas.SimpleTranslation
         output_field: translation_result   # → {"translation": "<pl>"}
         prompt:
           - role: user
             template: ${prompt_dir}/translate-simple.j2
   ```

Reusing the already-bound preset name means no `config/bindings/local.yaml`
edit (see [Configuration](Configuration)). The Polish string lands at
`translation_result.translation`. Run it on the vLLM box over the same
rows the pipeline processed:

```bash
uv run python run_pipeline.py experiment=translate_queries_trivia_simple
```

Output shards land under
`dataset/translate_queries_trivia_simple/00_translate/step-*.jsonl`.

## MetricX setup

MetricX is not on PyPI, and upstream pins a 2023-era stack
(`requirements.txt`: `transformers[torch]==4.30.2`,
`sentencepiece==0.1.99`, `datasets==2.13.1`) that predates this project's
`>=3.13` interpreter. The same version wall is why the scripts cannot
import the `DataCurator` package: everything runs from a 3.11 venv, and
the little project behaviour they need is reimplemented locally.

```
metricx/
├── .venv/               # 3.11 venv       (gitignored)
├── vendor/              # upstream clone  (gitignored)
├── reports/             # all output      (gitignored)
├── score_metricx.py
└── compare_metricx.py
```

```bash
git clone https://github.com/google-research/metricx metricx/vendor
uv venv metricx/.venv --python 3.11
uv pip install --python metricx/.venv \
    "transformers[torch]==4.30.2" sentencepiece==0.1.99 datasets==2.13.1 \
    "fsspec==2023.6.0" accelerate "numpy<2" "pyarrow<14" boto3 python-dotenv
```

`boto3` and `python-dotenv` are the scorer's own S3/env deps; the rest
are the upstream pins. Install the pins, not latest — `metricx23/models.py`
subclasses mT5 internals that newer `transformers` reshuffles. `datasets==2.13.1`
also leaves two transitive deps uncapped that its own code then breaks against:
`pyarrow` (imports `pa.PyExtensionType`, removed in ≥14 → pin `pyarrow<14`) and
`fsspec` (≥2023.9 trips `NotImplementedError: … LocalFileSystem …` on the local
JSONL → pin `fsspec==2023.6.0`). Skip
`mt-metrics-eval` (the fourth requirement): only their WMT eval scripts
need it, `predict` does not. Add `metricx/.venv/`, `metricx/vendor/` and
`metricx/reports/` to `.gitignore`. The scorer subprocesses
`sys.executable -m metricx23.predict` with `PYTHONPATH=metricx/vendor` —
same interpreter, no venv path to pass around — so the tested upstream
inference (input formatting, mT5 regression readout) is used verbatim. MetricX-QE XL (~3.7B) fits one H100 comfortably; sequence
it with vLLM so they don't contend for GPU memory.

## The generic scorer

`metricx/score_metricx.py` keeps the helper style
(`helpers/combine_experiments.py`: argparse, `pathlib`, type hints and
docstrings) but none of the helper imports — stdlib `json` instead of
`orjson`, and **no `DataCurator` imports**, so it runs under the 3.11
venv. One run scores one system.

```
metricx/.venv/bin/python metricx/score_metricx.py INPUT \
    --hyp-field FIELD [--source-field text] [--id-field id] \
    [--meta-field DOTTED ...] \
    --label NAME --out metricx/reports/NAME.jsonl \
    [--s3] [--env .env.browse] [--limit N] \
    [--metricx-model google/metricx-23-qe-xl-v2p0] \
    [--tokenizer google/mt5-xl] [--max-input-length 1024] [--batch-size 1] \
    [--cuda-device 0]
```

What it does:

- **Load.** With `--s3`, `INPUT` is an S3 prefix/key read with plain
  `boto3`: list the keys under the prefix, keep `*.jsonl`, fetch in key
  order — the `resolve_keys` / `read_all` behaviour of
  `DataCurator.distributed.s3io`, reimplemented in ~60 lines rather than
  imported (`--env` is loaded with `python-dotenv`). Without `--s3`, it
  is a local file / glob / stage directory
  (`step-*.jsonl`), read like `combine_experiments.load_input`. `--limit`
  caps *records* after loading (unlike `browse_jsonl_s3.py`, whose
  `--limit` caps S3 objects) — that is what the smoke runs below rely on.
- **Extract.** `--source-field` and the dotted `--hyp-field` are resolved
  with a local copy of `_resolve_path` (and its `_MISSING` sentinel) from
  `DataCurator/pipeline/modifiers/candidates.py` — 15 lines of `in`/`[]`,
  copied rather than imported, working on plain record dicts — letting
  nested keys like `translation_result.translation` resolve. Records are deduped by
  `--id-field`, keeping the first occurrence (S3 stage output is
  at-least-once; a re-emitted chunk can repeat ids). Rows with
  `error: true` or a missing/empty hypothesis are dropped and counted,
  per reason. Each repeatable `--meta-field` copies one extra dotted
  field into the report row (used below for the route breakdown).
- **Score.** Writes a temp `{"source","hypothesis"}` JSONL built **only**
  from the extracted fields — records are never passed through, which
  matters because the input rows already carry a top-level `source`
  meaning the *dataset* name ("trivia"), while MetricX's `source` must be
  the English text. Subprocesses `sys.executable -m metricx23.predict
  --qe` (with `PYTHONPATH=metricx/vendor`) and reads back the per-line
  `prediction`. 1024 is the MetricX-23 training max input length (1536
  belongs to MetricX-24; do not carry it over).
- **Report.** Writes `metricx/reports/<label>.jsonl`
  (`{id, source, hypothesis, metricx_qe}` + any meta fields) and a
  machine-readable `metricx/reports/<label>.summary.json` (n scored, n dropped by
  reason, mean/median/std/min/max, p10/p90), also printed. Remember:
  **lower is better**.

## Producing the reports

The `<sig>` in the finalize prefix is an opaque hash — discover the live
one first (see [Distributed](Distributed) for the S3 key layout):

```bash
uv run python helpers/browse_jsonl_s3.py --list \
    pipeline/gate_queries_trivia_no_thinking_batching2/stage/04_finalize/
```

Then score each system (pipeline and Google Translate share the S3 prefix):

```bash
# 1) full pipeline (S3) — carry the gate verdict for the route breakdown
metricx/.venv/bin/python metricx/score_metricx.py --s3 \
    pipeline/gate_queries_trivia_no_thinking_batching2/stage/04_finalize/<sig>/ \
    --hyp-field final_translation --meta-field gate_answer.verdict \
    --label pipeline --out metricx/reports/pipeline.jsonl

# 2) raw Google Translate (same S3 records, different field)
metricx/.venv/bin/python metricx/score_metricx.py --s3 \
    pipeline/gate_queries_trivia_no_thinking_batching2/stage/04_finalize/<sig>/ \
    --hyp-field text_gtranslate --label gtranslate \
    --out metricx/reports/gtranslate.jsonl

# 3) simple Qwen prompt (local baseline output)
metricx/.venv/bin/python metricx/score_metricx.py \
    dataset/translate_queries_trivia_simple/00_translate \
    --hyp-field translation_result.translation --label qwen_simple \
    --out metricx/reports/qwen_simple.jsonl
```

S3 credentials come from `.env.browse` (copy `.env.browse.example`; fill
`JOB_S3_ENDPOINT`, `JOB_S3_ACCESS_KEY_ID`, `JOB_S3_SECRET_ACCESS_KEY`,
`DC_S3_BUCKET`, `DC_S3_PREFIX=pipeline`) — the same file
`browse_jsonl_s3.py` reads, here loaded with `python-dotenv` directly.

## The final report

`metricx/compare_metricx.py` (same venv, no GPU; pure stdlib — `json`,
`statistics`, `csv`, `random` — pandas is not worth adding for a join
plus means) produces the final deliverable:

```bash
metricx/.venv/bin/python metricx/compare_metricx.py \
    metricx/reports/pipeline.jsonl metricx/reports/gtranslate.jsonl \
    metricx/reports/qwen_simple.jsonl \
    --baseline gtranslate --out metricx/reports/summary
```

It inner-joins the reports on `id` — every system is judged on the
identical row set — and derives a per-row **route** from the pipeline
report: `needs_edit` when `gate_answer.verdict` says so, else `clean`
when the pipeline and gtranslate hypotheses are byte-identical (the
finalize coalesce passed the raw draft through), else `minor_fix` (the
gate's minimal edit was used). It writes `metricx/reports/summary.csv`
(one row per id: route plus each system's score — the plotting-ready
table) and `metricx/reports/summary.md`, the report itself, with exactly
these sections:

1. **Run block** — date, git commit, MetricX model id, per-system
   pre-join n and drop counts by reason, joined n. Rows lost to the join
   are asymmetry worth a sentence, not silence.
2. **Headline table** — n, mean, median, std, p10/p90 per system, plus
   one sentence naming the better system and by how much (lower is
   better — say so in the table caption; every reader forgets).
3. **Pairwise vs the baseline** — for each system: mean paired delta,
   a 95% CI from a paired bootstrap over per-id deltas (10k resamples,
   stdlib `random.choices`), and win/tie/loss counts. Ties are *exact*
   score equality and genuinely common: on `clean` rows the pipeline
   output *is* the Google draft, so the (source, hypothesis) pair —
   and the score — are identical. Call a difference real only when the
   CI excludes zero.
4. **Route breakdown** — n and per-system mean per route. This is the
   section that answers the actual question: did the gate's edits help
   where it intervened (`needs_edit`, `minor_fix`), and did it do no
   harm where it passed the draft through (`clean`)?
5. **Score distribution** — per-system row counts in buckets
   `[0,1) [1,3) [3,6) [6,25]`; a markdown table beats an image here.
6. **Examples** — the 10 largest regressions and 10 largest improvements
   of the pipeline vs the baseline: id, English source, both hypotheses,
   both scores. Regressions first — they are the actionable list.
7. **Caveats** — reference-free QE is noisy at the segment level (trust
   aggregates, not single rows); en→pl is not a WMT23 language pair, so
   MetricX-23 ranks systems on the same data credibly but its absolute
   numbers are not calibrated quality claims; inputs truncate at 1024
   tokens (irrelevant for trivia queries, not for a future documents
   run); dropped-row counts per system.

## Verification

Smoke every piece on a handful of rows before scaling up:

1. **Baseline** — `run_pipeline.py experiment=translate_queries_trivia_simple
   dataset.limit=5`; confirm `translation_result.translation` holds real
   Polish text.
2. **S3 read** — `score_metricx.py --s3 <sig-prefix> --hyp-field
   final_translation --limit 5`; confirm chunks list, fields resolve, and
   `error:true` rows drop.
3. **Field path** — the same on the local Qwen output with
   `--hyp-field translation_result.translation --limit 5`.
4. **MetricX sanity** — scores land in [0,25]; a deliberately garbled
   hypothesis scores *higher* (worse) than a good one; the isolated venv's
   `metricx23.predict` loads the QE model on GPU.
5. **Combiner** — `compare_metricx.py` on the three 5-row reports renders
   and saves the joined table.
6. **Tie cross-check** — after the full pipeline and gtranslate runs,
   every `clean`-route id must have *identical* scores in both reports
   (same source, same hypothesis, deterministic model). A mismatch means
   the id join or the scoring is broken — this is the cheapest
   end-to-end correctness probe the data offers.

Only then run the three systems at full scale.

Notes:

- The 3.11 venv doubles as the standalone check: `DataCurator` cannot be
  installed there (`requires-python >= 3.13`), so any accidental project
  import fails at startup.
- Pass the **sig-qualified** finalize prefix; the bare `04_finalize/`
  prefix can include stale dead-sig chunks if GC has not run.
- Keep `--batch-size 1` (the scorer default, matching every upstream README
  example). Upstream `predict.py` tokenizes with `padding=False` and builds a
  `Trainer` with no padding collator, so `default_data_collator` tries to
  `torch.stack` unequal-length inputs and `RuntimeError`s at any size > 1. On a
  multi-GPU box the scorer also pins `CUDA_VISIBLE_DEVICES` to one device
  (`--cuda-device`, default 0): upstream floors `batch_size // device_count`,
  so batch size 1 across two GPUs becomes 0 (`batch_size should be a positive
  integer value, but got batch_size=0`). Pass `--cuda-device N` to pick which
  GPU (e.g. the one not running vLLM).
- If `metricx23.predict` errors on load despite the pinned
  `transformers`/`datasets`, a transitive dep the pins leave uncapped has
  drifted — usually `pyarrow` (`AttributeError: … PyExtensionType` when ≥14;
  pin `pyarrow<14`), `fsspec` (`NotImplementedError: … LocalFileSystem …` on
  the local JSONL; pin `fsspec==2023.6.0`), `numpy` (`_ARRAY_API not found` /
  "compiled with NumPy 1.x cannot run in NumPy 2"; keep `numpy<2`) or `torch`
  (pin e.g.
  `torch==2.4.*`). Reinstall the pins **together** in one pass (e.g.
  `uv pip install --python metricx/.venv "numpy<2" "pyarrow<14"`) so the
  resolver doesn't bump `numpy` back to 2.x — a later single-package install
  will. All **inside the isolated venv**; the main pipeline env stays
  untouched either way.
- The fast T5 tokenizer build hits protobuf's `transformers==4.30.2` vs
  protobuf-≥4 break (`TypeError: Descriptors cannot be created directly`). The
  scorer already sidesteps it by running `metricx23.predict` with
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (pure-Python backend, used
  only for the one-off tokenizer conversion — no inference cost); pinning
  `protobuf<3.21` in the venv is the equivalent alternative.
