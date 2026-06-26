# Running on the no Docker box

Instructions for whoever runs the pipeline on the no docker system. LanguageTool is
**not** hosted here — it runs elsewhere and is reached over a public domain, so
this box only needs vLLM + the pipeline.

# 0. Setup
```bash
uv sync
```

# 1. Download dataset
```bash
uv run python ./helpers/HF_mined_ds_to_jsonl.py --dataset jansowa/trivia-mined-negatives --source trivia --out-prefix trivia_mined
```

# 2. Run vLLM + adjust parameters
Runs natively (no Docker). Set `--tensor-parallel-size` to the number of H100s.
```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --tensor-parallel-size 2 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only --gpu-memory-utilization 0.95 --max-num-seqs 100 --max-num-batched-tokens 4096 --enable-prefix-caching --kv-cache-dtype fp8
```

# 3. Adjust parameters based on the hardware
## ./config/config.yaml
- runner.concurrency

## ./config/model/qwen35_stage1_no_thinking.yaml
- api_base
- concurrency

## ./config/model/qwen35_stage2.yaml
- api_base
- concurrency

## ./config/experiment/<your-experiment>.yaml
- remote_server — public LanguageTool URL (hosted on another machine)

## Smoke test: LanguageTool reachability
Confirm the public server answers before launching a long run (use the
`remote_server` value you set above):
```bash
curl --fail --get http://languagetool-server-pc1.rqlabs.space/v2/check \
  --data-urlencode "language=pl-PL" \
  --data-urlencode "text=To jest test ze bledem"
```
A reachable server returns HTTP 200 with JSON (`"matches": [...]`). A non-zero
exit or connection error means fix the URL/hosting before running the pipeline.

# 4. Start the runner
```bash
uv run python run_pipeline.py
```