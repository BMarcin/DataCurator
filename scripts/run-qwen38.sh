#!/usr/bin/env bash
# Qwen3.8-27B-FP8, TP=2 over the 4090 + 5090. Reasoning is switched OFF per request by
# the qwen38_* presets (enable_thinking: false), not by a server flag -- the chat
# template answers that kwarg by pre-emitting a closed "<think>\n\n</think>" block, so
# the whole generation falls inside the guided-decoding region and `response_format`
# actually constrains it. (That is the property Muse-Glimmer lacked.)
#
# Model choice is forced by VRAM: the FP8 repo is 30.9 GiB of weights and the 5090
# holds 31.8 GiB, so it only fits sharded across both cards. The 4090 is the binding
# card -- TP shards evenly, so the 5090's extra 8 GiB goes unused.
#
# --language-model-only drops the (unquantized, BF16) vision tower; the repo is a
# Qwen3_5ForConditionalGeneration and we only ever send text.
# VLLM_USE_DEEP_GEMM=0: the weights are block-quantized FP8 (128x128) and DeepGEMM
# needs SM90+, but the 4090 is SM89 -- force the portable Triton path for the whole TP
# group (same reason scripts/run-muse-glimmer.sh sets it).
# Needs vLLM >= 0.17 / transformers >= 5.8 for the qwen3_5 hybrid architecture.
# HF_TOKEN is optional (Apache-2.0, public repo); export it for your own rate limits.

docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    --env "VLLM_USE_DEEP_GEMM=0" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:cu130-nightly \
    Qwen/Qwen3.8-27B-FP8 --port 8000 \
    --tensor-parallel-size 2 --max-model-len 90000 \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --language-model-only \
    --gpu-memory-utilization 0.94 --max-num-seqs 20 \
    --enable-prefix-caching --kv-cache-dtype fp8

# --max-num-seqs is 24 rather than the 35 run-muse-glimmer.sh serves because this model
# is hybrid: 48 of its 64 layers are Gated DeltaNet, whose recurrent state is per
# SEQUENCE, not per token (~144 MiB/seq, ~72 MiB per rank at TP=2). Sequence slots
# therefore cost VRAM up front, before a single token of KV cache is allocated. Only 16
# layers keep a real KV cache, so raising --max-model-len is cheap and raising
# --max-num-seqs is not.
#
# If it OOMs, the 4090 is the card that runs out. In rough order of what to try:
#   1. drop --gpu-memory-utilization to 0.90
#   2. drop --max-num-seqs to 16     (keep the qwen38_* presets' `concurrency` below
#                                     whatever you set)
#   3. drop --max-model-len          (only together with the gate's max_tokens: 80000 in
#                                     config/experiment/gate_queries_trivia_qwen38_batching.yaml
#                                     -- vLLM rejects a request whose prompt + max_tokens
#                                     exceeds it)
# If TP=2 hangs at startup instead, add --env "NCCL_P2P_DISABLE=1" (Ada<->Blackwell PCIe P2P).
# If the container reports no CUDA device at all (while nvidia-smi on the host is fine),
# regenerate /etc/cdi/nvidia.yaml with nvidia-ctk -- the spec hardcodes the dynamic
# nvidia-uvm device major, which changes across reboots.
#
# No MTP speculative decoding, even though the repo ships mtp.safetensors and the vLLM
# recipe suggests --speculative-config '{"method":"mtp","num_speculative_tokens":3}':
# the drafter's weights plus the rejection sampler's activations are exactly what OOM'd
# the 4090 mid-job on the 2026-08-18 muse run.
