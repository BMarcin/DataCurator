"""Hydra entrypoint for the DataCurator pipeline.

Run the default experiment:

    uv run python run_pipeline.py

Pick an experiment and enable notifications:

    uv run python run_pipeline.py experiment=example notifications=ntfy

Publish live progress to the dashboard S3 bucket:

    uv run python run_pipeline.py reporting=s3

Resume from a specific stage, or disable one on the fly:

    uv run python run_pipeline.py runner.start_from=punctuation
    uv run python run_pipeline.py pipeline.whitespace.enabled=false
"""
from __future__ import annotations

import asyncio
import logging
import sys

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from DataCurator.pipeline.runner import PipelineRunner, build_pipeline
from DataCurator.reporting import build_reporter

load_dotenv()

# Third-party libraries that log via the stdlib ``logging`` module (httpx logs
# every HTTP request at INFO). Quieted to WARNING unless ``runner.debug`` is on.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "requests")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the pipeline from ``cfg`` and run it to completion."""
    # `runner.debug` drives the log level: DEBUG when on, INFO otherwise.
    debug = bool(cfg.runner.get("debug", False))
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    # Tame chatty stdlib loggers (e.g. httpx's per-request INFO lines), which
    # loguru's level does not govern.
    noisy_level = logging.DEBUG if debug else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    specs = build_pipeline(cfg)
    experiment_name = cfg.get("experiment_name", "pipeline")
    reporter = build_reporter(
        cfg.get("reporting"), name=experiment_name, job_cfg=cfg.get("job")
    )
    runner = PipelineRunner(
        specs,
        name=experiment_name,
        dataset_cfg=cfg.dataset,
        runner_cfg=cfg.runner,
        notifications_cfg=cfg.get("notifications"),
        reporter=reporter,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
