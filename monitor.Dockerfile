# Monitor (run_monitor.py). Build context = repo root.
#
# The monitor never instantiates stages, so it needs neither config/ nor prompt
# assets. It reads each job's version from the job record, so unlike the worker
# it has no DC_VERSION. Connection settings are provided at runtime.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY DataCurator ./DataCurator
COPY run_monitor.py ./
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --uid 1000 --no-create-home monitor
USER monitor

CMD ["python", "run_monitor.py"]
