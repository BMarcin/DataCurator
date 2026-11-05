# Distributed worker (run_distributed.py). Build context = repo root.
#
# Behaviour and prompts arrive in the work messages, so the image needs neither
# config/ nor prompt assets. NATS and S3 settings and the endpoint bindings are
# provided at runtime, never built in.
#
# DC_VERSION is required because the image carries no .git. CI passes the release
# tag. A worker only pulls jobs stamped with its own version.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY DataCurator ./DataCurator
COPY run_distributed.py ./

ARG DC_VERSION
RUN test -n "$DC_VERSION" || { echo "build with --build-arg DC_VERSION=<release tag>" >&2; exit 1; }
ENV DC_VERSION=$DC_VERSION
ENV PATH="/app/.venv/bin:$PATH"

# The worker writes rotating log files to logs/ under its working directory.
RUN useradd --uid 1000 --no-create-home worker && mkdir logs && chown worker logs
USER worker

# Exec form: Python is PID 1 and receives docker stop's SIGTERM, which drains it.
CMD ["python", "run_distributed.py"]
