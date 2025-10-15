"""
Performance profiling utilities for identifying bottlenecks.

Provides decorators and context managers for timing operations,
profiling functions, and identifying performance bottlenecks.
"""

import time
import functools
import torch
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Median execution time."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]


class PerformanceProfiler:
    """
    Performance profiler for tracking operation timings.

    Thread-safe profiler that collects timing statistics for operations
    and provides analysis of performance bottlenecks.

    Example:
        >>> profiler = PerformanceProfiler()
        >>>
        >>> # Using context manager
        >>> with profiler.profile("data_loading"):
        ...     data = load_data()
        >>>
        >>> # Using decorator
        >>> @profiler.profile_function
        ... def my_function():
        ...     pass
        >>>
        >>> # Get report
        >>> print(profiler.get_report())
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.stats: Dict[str, TimingStats] = {}
        self.lock = threading.Lock()
        self._cuda_available = torch.cuda.is_available()

    def record_timing(self, name: str, elapsed_time: float):
        """
        Record timing for an operation.

        Args:
            name: Operation name
            elapsed_time: Elapsed time in seconds
        """
        if not self.enabled:
            return

        with self.lock:
            if name not in self.stats:
                self.stats[name] = TimingStats(name=name)

            stats = self.stats[name]
            stats.count += 1
            stats.total_time += elapsed_time
            stats.min_time = min(stats.min_time, elapsed_time)
            stats.max_time = max(stats.max_time, elapsed_time)
            stats.times.append(elapsed_time)

            # Keep only last 1000 times to avoid memory issues
            if len(stats.times) > 1000:
                stats.times = stats.times[-1000:]

    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Args:
            name: Operation name

        Returns:
            Context manager

        Example:
            >>> with profiler.profile("forward_pass"):
            ...     output = model(input)
        """
        return ProfilingContext(self, name)

    def profile_function(self, func: Optional[Callable] = None, name: Optional[str] = None):
        """
        Decorator for profiling a function.

        Args:
            func: Function to profile
            name: Optional custom name

        Returns:
            Decorated function

        Example:
            >>> @profiler.profile_function
            ... def my_function():
            ...     pass
            >>>
            >>> @profiler.profile_function(name="custom_name")
            ... def another_function():
            ...     pass
        """
        def decorator(f: Callable) -> Callable:
            operation_name = name or f.__name__

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                with self.profile(operation_name):
                    return f(*args, **kwargs)
            return wrapper

        # Handle both @profile_function and @profile_function()
        if func is None:
            return decorator
        return decorator(func)

    def get_stats(self) -> Dict[str, TimingStats]:
        """
        Get all timing statistics.

        Returns:
            Dictionary of timing statistics
        """
        with self.lock:
            return dict(self.stats)

    def get_report(self, top_n: int = 20, sort_by: str = "total") -> str:
        """
        Generate performance report.

        Args:
            top_n: Number of top operations to show
            sort_by: Sort key ("total", "avg", "count", "max")

        Returns:
            Formatted report string
        """
        stats_list = list(self.get_stats().values())

        if not stats_list:
            return "No profiling data available."

        # Sort by specified metric
        sort_keys = {
            "total": lambda s: s.total_time,
            "avg": lambda s: s.avg_time,
            "count": lambda s: s.count,
            "max": lambda s: s.max_time,
        }
        sort_key = sort_keys.get(sort_by, sort_keys["total"])
        stats_list.sort(key=sort_key, reverse=True)

        # Limit to top N
        stats_list = stats_list[:top_n]

        # Build report
        lines = ["=" * 100]
        lines.append("Performance Profiling Report")
        lines.append("=" * 100)
        lines.append("")

        # Header
        header = f"{'Operation':<40} {'Count':>8} {'Total(s)':>10} {'Avg(ms)':>10} {'Med(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
        lines.append(header)
        lines.append("-" * 100)

        # Data rows
        for stats in stats_list:
            row = (
                f"{stats.name:<40} "
                f"{stats.count:>8} "
                f"{stats.total_time:>10.3f} "
                f"{stats.avg_time * 1000:>10.2f} "
                f"{stats.median_time * 1000:>10.2f} "
                f"{stats.min_time * 1000:>10.2f} "
                f"{stats.max_time * 1000:>10.2f}"
            )
            lines.append(row)

        lines.append("=" * 100)

        # Summary
        total_time = sum(s.total_time for s in stats_list)
        total_calls = sum(s.count for s in stats_list)
        lines.append("")
        lines.append(f"Total time: {total_time:.3f}s")
        lines.append(f"Total calls: {total_calls}")
        lines.append("")

        return "\n".join(lines)

    def get_bottlenecks(self, threshold_percent: float = 5.0) -> List[TimingStats]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percent: Minimum percentage of total time to be considered bottleneck

        Returns:
            List of bottleneck operations
        """
        stats_list = list(self.get_stats().values())

        if not stats_list:
            return []

        total_time = sum(s.total_time for s in stats_list)

        bottlenecks = [
            s for s in stats_list
            if (s.total_time / total_time * 100) >= threshold_percent
        ]

        bottlenecks.sort(key=lambda s: s.total_time, reverse=True)

        return bottlenecks

    def reset(self):
        """Reset all statistics."""
        with self.lock:
            self.stats.clear()

    def save_report(self, output_path: str, **kwargs):
        """
        Save performance report to file.

        Args:
            output_path: Output file path
            **kwargs: Additional arguments for get_report()
        """
        report = self.get_report(**kwargs)

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Performance report saved to {output_path}")


class ProfilingContext:
    """Context manager for profiling."""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.cuda_available = torch.cuda.is_available()

    def __enter__(self):
        """Enter context."""
        if not self.profiler.enabled:
            return self

        # Synchronize CUDA if available
        if self.cuda_available:
            torch.cuda.synchronize()

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if not self.profiler.enabled or self.start_time is None:
            return False

        # Synchronize CUDA if available
        if self.cuda_available:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self.start_time
        self.profiler.record_timing(self.name, elapsed)

        return False  # Don't suppress exceptions


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """
    Get or create global profiler instance.

    Returns:
        Global profiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(name: str):
    """
    Context manager using global profiler.

    Args:
        name: Operation name

    Returns:
        Context manager

    Example:
        >>> with profile("data_loading"):
        ...     data = load_data()
    """
    return get_global_profiler().profile(name)


def profile_function(func: Optional[Callable] = None, name: Optional[str] = None):
    """
    Decorator using global profiler.

    Args:
        func: Function to profile
        name: Optional custom name

    Returns:
        Decorated function

    Example:
        >>> @profile_function
        ... def my_function():
        ...     pass
    """
    return get_global_profiler().profile_function(func, name)


def get_performance_report(**kwargs) -> str:
    """
    Get performance report from global profiler.

    Args:
        **kwargs: Arguments for get_report()

    Returns:
        Formatted report string
    """
    return get_global_profiler().get_report(**kwargs)


def identify_bottlenecks(threshold_percent: float = 5.0) -> List[TimingStats]:
    """
    Identify bottlenecks using global profiler.

    Args:
        threshold_percent: Minimum percentage threshold

    Returns:
        List of bottleneck operations
    """
    return get_global_profiler().get_bottlenecks(threshold_percent)