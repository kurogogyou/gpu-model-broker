"""GPU + CPU observability helpers.

Decision per 2026-06-13 (see gpu-model-broker.md `## Decisions`):
- VRAM via nvidia-smi (NOT torch.cuda.memory_allocated — CTranslate2 allocates
  outside torch's tracking, so torch reports 5 MiB for whisperx vs ~2 GiB real).
- CPU RSS via /proc/<container_main_pid>/status VmRSS line. Observability only;
  no eviction on CPU RSS in v1.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


_VMRSS_RE = re.compile(r"^VmRSS:\s+(\d+)\s+kB", re.MULTILINE)


def gpu_used_mb() -> int:
    """Total VRAM currently in use on GPU 0. Includes host overhead + all
    containers + any other process on the GPU."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return int(out.stdout.strip())


def gpu_total_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    return int(out.stdout.strip())


def container_cpu_rss_mb(pid: int) -> int | None:
    """Container main process RSS in MiB. Returns None if PID is gone."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    m = _VMRSS_RE.search(status)
    if not m:
        return None
    return int(m.group(1)) // 1024
