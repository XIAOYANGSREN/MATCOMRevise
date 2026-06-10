"""Timing helpers and the hardware report written next to the timing results."""

import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimeRecord:
    label: str
    wall_seconds: float
    cpu_seconds: float
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "label": self.label,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            **self.meta,
        }


class Timer:
    """Context-manager timer; synchronises CUDA around the measurement."""

    def __init__(self, label="block", sync_cuda=True):
        self.label = label
        self.sync_cuda = sync_cuda
        self.record = None
        self._t0_wall = 0.0
        self._t0_cpu = 0.0

    def _cuda_sync(self):
        if not self.sync_cuda:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def __enter__(self):
        self._cuda_sync()
        self._t0_wall = time.perf_counter()
        self._t0_cpu = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cuda_sync()
        wall = time.perf_counter() - self._t0_wall
        cpu = time.process_time() - self._t0_cpu
        self.record = TimeRecord(label=self.label, wall_seconds=wall, cpu_seconds=cpu)


@contextmanager
def timed(label="block", sync_cuda=True):
    t = Timer(label=label, sync_cuda=sync_cuda)
    with t:
        yield t


def time_call(label, fn, *args, **kwargs):
    with Timer(label) as t:
        result = fn(*args, **kwargs)
    return result, t.record


def _safe(callable_, default="n/a"):
    try:
        return str(callable_())
    except Exception:
        return default


def collect_hardware_info():
    info = {
        "hostname": _safe(socket.gethostname),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": sys.version.replace("\n", " "),
        "cpu_count_logical": os.cpu_count(),
    }

    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["memory_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        info["cpu_count_physical"] = "n/a"
        info["memory_total_gb"] = "n/a"

    info["cpu_model"] = platform.processor() or "n/a"
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"], stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "Name" not in ln]
            if lines:
                info["cpu_model"] = lines[0]
        except Exception:
            pass

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_models"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            try:
                info["gpu_memory_gb"] = [
                    round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
                    for i in range(torch.cuda.device_count())
                ]
            except Exception:
                info["gpu_memory_gb"] = "n/a"
        else:
            info["cuda_version"] = None
            info["gpu_count"] = 0
    except Exception:
        info["torch"] = "not installed"
        info["cuda_available"] = False
        info["cuda_version"] = None
        info["gpu_count"] = 0

    for pkg in ("numpy", "scipy", "matplotlib", "pandas", "tqdm"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "?")
        except Exception:
            info[f"{pkg}_version"] = "not installed"

    return info


def format_hardware_report(info):
    keys_order = [
        "hostname", "os", "machine", "cpu_model", "cpu_count_physical",
        "cpu_count_logical", "memory_total_gb", "torch", "cuda_available",
        "cuda_version", "gpu_count", "gpu_models", "gpu_memory_gb", "python",
        "numpy_version", "scipy_version", "matplotlib_version",
        "pandas_version", "tqdm_version",
    ]
    lines = ["Hardware / Software Environment", "=" * 60]
    for k in keys_order:
        if k in info:
            lines.append(f"{k:<22}: {info[k]}")
    return "\n".join(lines)


def save_hardware_report(path):
    info = collect_hardware_info()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_hardware_report(info))
        f.write("\n")
    return info


if __name__ == "__main__":
    print(format_hardware_report(collect_hardware_info()))
