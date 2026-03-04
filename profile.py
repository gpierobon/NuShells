import time
import functools
from collections import defaultdict

# Global registry
_timings: dict[str, list[float]] = defaultdict(list)


def timed(label: str = None):
    """
    Decorator that records wall-clock time for each call.
    Usage:
        @timed()                  # uses 'ClassName.method_name' automatically
        @timed("custom label")    # explicit label
    """
    def decorator(fn):
        key = label or f"{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            _timings[key].append(time.perf_counter() - t0)
            return result

        return wrapper
    return decorator


def report(unit: str = "ms", top: int = None):
    """
    Print a summary table of all recorded timings.

    Args:
        unit:  's' | 'ms' | 'us'
        top:   if set, only show the N slowest entries (by total time)
    """
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]
    sym   = {"s": "s",  "ms": "ms", "us": "μs"}[unit]

    rows = []
    for key, times in _timings.items():
        n     = len(times)
        total = sum(times) * scale
        rows.append((key, n, total))

    rows.sort(key=lambda r: r[2], reverse=True)          # sort by total time
    if top:
        rows = rows[:top]

    grand_total = sum(r[2] for r in rows)

    col_w = max(len(r[0]) for r in rows) + 2
    header = (
        f"{'Function':<{col_w}} {'Calls':>6}  "
        f"{'Total':>10}  {'%':>6}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(f"  Timing report  [{sym}]")
    print(sep)
    print(header)
    print(sep)
    for key, n, total in rows:
        pct = 100.0 * total / grand_total if grand_total > 0 else 0.0
        print(
            f"{key:<{col_w}} {n:>6}  "
            f"{total:>10.3f}  {pct:>5.1f}%"
        )
    print(sep + "\n")


def reset():
    """Clear all recorded timings."""
    _timings.clear()
