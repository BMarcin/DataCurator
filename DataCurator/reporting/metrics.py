"""Best-effort host metrics sampling (CPU / RAM / GPU).

These feed the ``metrics`` block of the status envelope so the dashboard can
show live machine load next to job progress. Everything here is best-effort:
if ``psutil`` misbehaves or no NVIDIA GPU/driver is present, the relevant
piece is simply omitted — sampling never raises into the caller.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

try:  # psutil is a hard dep, but guard anyway so a broken install can't abort a run.
    import psutil
except Exception:  # noqa: BLE001 - any import failure means "no CPU/RAM metrics".
    psutil = None  # type: ignore[assignment]

# nvidia-ml-py exposes the ``pynvml`` module. Absent driver/library => no GPU block.
try:
    import pynvml  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore[assignment]

_nvml_ready: Optional[bool] = None  # tri-state: None=untried, True/False=init result


def _ensure_nvml() -> bool:
    """Initialise NVML once; return whether GPU sampling is available."""
    global _nvml_ready
    if _nvml_ready is not None:
        return _nvml_ready
    if pynvml is None:
        _nvml_ready = False
        return False
    try:
        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception as exc:  # noqa: BLE001 - no usable GPU; degrade silently.
        logger.debug(f"GPU metrics unavailable (NVML init failed): {exc}")
        _nvml_ready = False
    return _nvml_ready


def gpu_names() -> List[str]:
    """Return the model name of each visible GPU (empty if none/unavailable)."""
    if not _ensure_nvml():
        return []
    names: List[str] = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            names.append(name.decode() if isinstance(name, bytes) else str(name))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"GPU name query failed: {exc}")
    return names


def _sample_gpus() -> List[Dict[str, Any]]:
    """Sample per-GPU utilisation/memory/temperature/power, best-effort."""
    if not _ensure_nvml():
        return []
    gpus: List[Dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:  # noqa: BLE001
        return []
    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            entry: Dict[str, Any] = {
                "index": i,
                "util_pct": int(util.gpu),
                "mem_pct": round(mem.used / mem.total * 100, 1) if mem.total else None,
            }
            try:
                entry["temp_c"] = int(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:  # noqa: BLE001 - temperature is optional.
                pass
            try:
                entry["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            except Exception:  # noqa: BLE001 - power draw is optional.
                pass
            gpus.append({k: v for k, v in entry.items() if v is not None})
        except Exception as exc:  # noqa: BLE001 - skip a flaky device, keep the rest.
            logger.debug(f"GPU {i} sample failed: {exc}")
    return gpus


def sample_system_metrics(*, sample_gpu: bool = True) -> Optional[Dict[str, Any]]:
    """Return a ``metrics`` snapshot, or ``None`` if nothing could be sampled.

    ``cpu_pct`` is measured since the previous call (``psutil`` semantics for
    ``interval=None``), so calling this on a fixed cadence yields meaningful
    deltas. The GPU block is omitted when no NVIDIA device is available.
    """
    metrics: Dict[str, Any] = {}
    if psutil is not None:
        try:
            metrics["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
            metrics["ram_pct"] = round(psutil.virtual_memory().percent, 1)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"CPU/RAM metrics sample failed: {exc}")
    if sample_gpu:
        gpus = _sample_gpus()
        if gpus:
            metrics["gpu"] = gpus
    return metrics or None
