FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# CUDA build of torch (also runs on CPU). cu128 because the deploy box's
# RTX 5090 is sm_120 (Blackwell) — cu121 wheels ship no kernels for it.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY batcher.py metricx_model.py server.py variants.py ./

EXPOSE 8000
# Weights are NOT baked in (~15GB) — they load into the mounted HF_HOME cache
# on first start; /health returns 200 only once loading completes.
CMD ["python", "server.py"]
