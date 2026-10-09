"""Independent, reproducible baseline benchmarks for the ASD project.

This package deliberately does not import or alter the project's training
pipeline.  Run it with ``python -m benchmarks`` from the project directory.
"""

from .config import BenchmarkConfig
from .engine import AVAILABLE_BASELINES, run_benchmarks

__all__ = ["AVAILABLE_BASELINES", "BenchmarkConfig", "run_benchmarks"]
