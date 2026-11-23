"""Utilitários de profiling de tempo e memória em nível de subrotina.

Observação:
    O módulo `tracemalloc` do Python introduz overhead de CPU (tipicamente 2x a 5x)
    por interceptar alocações de memória. Para não distorcer as curvas empíricas de
    complexidade temporal, este módulo separa explicitamente a medição de tempo
    limpo (`measure_memory=False`) da medição de pico de memória (`measure_memory=True`).
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SubroutineMetrics:
    time_s: float = 0.0
    peak_mem_kb: float = 0.0


@dataclass
class ProfileCollector:
    """Acumula métricas de tempo e memória por subrotina."""

    subroutines: Dict[str, SubroutineMetrics] = field(default_factory=dict)

    def record(self, name: str, time_s: float, peak_mem_kb: float = 0.0) -> None:
        if name not in self.subroutines:
            self.subroutines[name] = SubroutineMetrics(time_s=time_s, peak_mem_kb=peak_mem_kb)
        else:
            self.subroutines[name].time_s += time_s
            self.subroutines[name].peak_mem_kb = max(self.subroutines[name].peak_mem_kb, peak_mem_kb)

    def to_flat_dict(self, prefix: str = "sub_") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, m in self.subroutines.items():
            out[f"{prefix}{name}_time_s"] = m.time_s
            out[f"{prefix}{name}_peak_kb"] = m.peak_mem_kb
        return out


@contextmanager
def profile_subroutine(
    collector: ProfileCollector | None,
    name: str,
    measure_memory: bool = False,
):
    """Context manager para instrumentar subrotinas individuais."""
    started_here = False
    if measure_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            started_here = True
        else:
            tracemalloc.reset_peak()

    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        peak_kb = 0.0
        if measure_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
            peak_kb = peak_bytes / 1024.0
            if started_here:
                tracemalloc.stop()
        if collector is not None:
            collector.record(name, dt, peak_kb)


def measure_peak_memory_kb(func, *args, **kwargs):
    """Executa uma função isoladamente medindo apenas o pico de memória alocada em KB."""
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return result, peak_bytes / 1024.0