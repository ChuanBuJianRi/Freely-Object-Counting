"""GPU thermal-safety guard shared by all GOC evaluators / launchers.

Project policy (see AGENTREAD.md): every run that touches the GPU MUST go
through ``GpuGuard`` so we never bake the card. The guard:

- polls ``nvidia-smi`` every ``check_every`` images;
- when temp >= ``temp_limit`` C, ``torch.cuda.synchronize()`` + sleep
  ``cooldown_sec`` and then keep sleeping in 10s slices until temp drops
  below ``temp_limit - hysteresis``;
- records peak temp + total cool-down seconds + number of cool-down events
  into the run's metrics.json (``thermal: {...}``).

The ``nvidia-smi`` binary path is auto-detected (covers WSL2 + native Linux);
if it cannot be found, the guard turns into a no-op and prints a one-line
warning so CPU-only experiments keep working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


_NVIDIA_SMI_CANDIDATES = (
    "nvidia-smi",                       # in PATH (native Linux, most common)
    "/usr/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",      # WSL2
    "/mnt/c/Windows/System32/nvidia-smi.exe",  # WSL fallback
)


def _find_nvidia_smi() -> Optional[str]:
    """Find a usable ``nvidia-smi`` binary, or return None."""
    which = shutil.which("nvidia-smi")
    if which:
        return which
    for candidate in _NVIDIA_SMI_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
    return None


def query_temp(nvidia_smi: str, gpu_index: int = 0) -> Optional[int]:
    """Return the GPU temperature in Celsius, or None if the query failed."""
    try:
        out = subprocess.check_output(
            [nvidia_smi,
             f"--id={gpu_index}",
             "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


@dataclass
class GpuGuardStats:
    peak_temp_c: int = 0
    cooldown_events: int = 0
    cooldown_seconds: float = 0.0
    polls: int = 0


@dataclass
class GpuGuard:
    """Thermal guard for long-running GPU evaluation loops.

    Project default :: ``temp_limit=78``, ``cooldown_sec=30``,
    ``hysteresis=5``, ``check_every=5``. Override per-run via CLI flags
    (every evaluator script under ``codes/eval/`` exposes them).

    Set ``enabled=False`` to opt out (only allowed for CPU-only runs;
    document the reason in the run's README).
    """

    enabled: bool = True
    temp_limit: int = 78
    cooldown_sec: int = 30
    hysteresis: int = 5
    check_every: int = 5
    gpu_index: int = 0
    nvidia_smi: Optional[str] = None
    sync_cuda: bool = True

    stats: GpuGuardStats = field(default_factory=GpuGuardStats)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.nvidia_smi is None:
            self.nvidia_smi = _find_nvidia_smi()
        if self.nvidia_smi is None:
            print("[gpu-guard] nvidia-smi not found — guard disabled "
                  "(run will proceed without thermal protection).")
            self.enabled = False
            return
        first = query_temp(self.nvidia_smi, self.gpu_index)
        if first is None:
            print(f"[gpu-guard] nvidia-smi at {self.nvidia_smi} did not return a "
                  "temperature — guard disabled.")
            self.enabled = False
            return
        self.stats.peak_temp_c = first
        print(f"[gpu-guard] enabled  limit={self.temp_limit}C  "
              f"cooldown={self.cooldown_sec}s  hysteresis={self.hysteresis}C  "
              f"check_every={self.check_every}  gpu={self.gpu_index}  "
              f"start_temp={first}C  bin={self.nvidia_smi}")

    def maybe_throttle(self, idx: int = 0, total: int = 0) -> None:
        """Call this every image (cheap; only polls every ``check_every``)."""
        if not self.enabled:
            return
        if self.check_every > 1 and idx % self.check_every != 0:
            return

        temp = query_temp(self.nvidia_smi, self.gpu_index)
        self.stats.polls += 1
        if temp is None:
            return
        if temp > self.stats.peak_temp_c:
            self.stats.peak_temp_c = temp
        if temp < self.temp_limit:
            return

        if self.sync_cuda:
            try:
                import torch  # local import: keep CPU-only callers light
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

        self.stats.cooldown_events += 1
        print(f"  [gpu-guard {idx:4d}/{total}] temp={temp}C "
              f">= limit={self.temp_limit}C  -> sleeping {self.cooldown_sec}s")
        cool_start = time.time()
        time.sleep(self.cooldown_sec)
        target = self.temp_limit - self.hysteresis
        while True:
            t = query_temp(self.nvidia_smi, self.gpu_index)
            if t is None or t < target:
                break
            time.sleep(10)
        self.stats.cooldown_seconds += time.time() - cool_start
        post = query_temp(self.nvidia_smi, self.gpu_index)
        print(f"  [gpu-guard] resumed at {post}C "
              f"(cooled {time.time() - cool_start:.0f}s, "
              f"events={self.stats.cooldown_events}, "
              f"total_cool={self.stats.cooldown_seconds:.0f}s)")

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "temp_limit_c": self.temp_limit,
            "cooldown_sec": self.cooldown_sec,
            "hysteresis_c": self.hysteresis,
            "check_every": self.check_every,
            "peak_temp_c": self.stats.peak_temp_c,
            "cooldown_events": self.stats.cooldown_events,
            "cooldown_seconds": round(self.stats.cooldown_seconds, 1),
            "polls": self.stats.polls,
        }


def add_cli_args(parser) -> None:
    """Attach standard ``--gpu-*`` flags to an argparse parser.

    All GOC evaluators must expose this so launcher scripts can override
    the guard uniformly. Mirrors the field names of ``GpuGuard``.
    """
    g = parser.add_argument_group("GPU thermal safety (mandatory for GPU runs)")
    g.add_argument("--gpu-temp-limit", type=int, default=78,
                   help="Throttle when GPU temp >= this many degC. Default 78.")
    g.add_argument("--gpu-cooldown-sec", type=int, default=30,
                   help="Initial sleep on a hot event. Default 30s.")
    g.add_argument("--gpu-hysteresis", type=int, default=5,
                   help="Resume only after temp < limit - hysteresis. Default 5C.")
    g.add_argument("--gpu-check-every", type=int, default=5,
                   help="Poll temp every N images. Default 5.")
    g.add_argument("--gpu-index", type=int, default=0,
                   help="Which GPU to monitor (default 0).")
    g.add_argument("--gpu-guard-off", action="store_true",
                   help="Disable thermal guard. Only allowed for CPU-only runs; "
                        "must be justified in the run README.")


def guard_from_args(args) -> GpuGuard:
    """Build a GpuGuard from argparse Namespace produced by add_cli_args."""
    return GpuGuard(
        enabled=not args.gpu_guard_off,
        temp_limit=args.gpu_temp_limit,
        cooldown_sec=args.gpu_cooldown_sec,
        hysteresis=args.gpu_hysteresis,
        check_every=args.gpu_check_every,
        gpu_index=args.gpu_index,
    )
