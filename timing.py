import time
import functools
from collections import defaultdict

# Global registry
_timings: dict[str, list[float]] = defaultdict(list)
_children: dict[str, set[str]]   = defaultdict(set)
_wall_start: float = None
_wall_end:   float = None

def start_wall():
    global _wall_start
    _wall_start = time.perf_counter()

def stop_wall():
    global _wall_end
    _wall_end = time.perf_counter()


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


def report(unit: str = "ms", path: str = None, show: bool = True):
    """
    Print a summary table of all recorded timings.

    Args:
        unit:  's' | 'ms' | 'us'
        top:   if set, only show the N slowest entries (by total time)
        path:  if set, save the report to this file
    """
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]
    sym   = {"s": "s",  "ms": "ms", "us": "μs"}[unit]

    totals = {key: sum(times) for key, times in _timings.items()}
    counts = {key: len(times) for key, times in _timings.items()}

    if _wall_start is not None and _wall_end is not None:
        wall_total = _wall_end - _wall_start
    else:
        wall_total = max(totals.values()) if totals else 1.0

    def unaccounted(key):
        child_total = sum(totals.get(c, 0.0) for c in _children.get(key, set()))
        return totals[key] - child_total

    rows = []
    for key in totals:
        n       = counts[key]
        total_s = totals[key]
        pct     = 100.0 * total_s / wall_total
        unaccnt = unaccounted(key)
        rows.append((key, n, total_s, pct, unaccnt))

    rows.sort(key=lambda r: r[2], reverse=True)

    col_w  = max(len(r[0]) for r in rows) + 2
    header = (
        f"{'Function':<{col_w}} {'Calls':>7}  "
        f"{'Total':>10}  {'Wall%':>6}"
    )
    sep = "-" * len(header)

    lines = []
    lines.append(sep)
    lines.append(f"  Timing report  [{sym}]")
    if _wall_start is not None:
        lines.append(f"  Wall total: {wall_total*scale:.3f} {sym}")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for key, n, total_s, pct, unaccnt in rows:
        lines.append(
            f"{key:<{col_w}} {n:>7}  "
            f"{total_s*scale:>10.3f}  {pct:>5.1f}%  "
        )

    if _wall_start is not None:
        all_children   = set(c for cs in _children.values() for c in cs)
        top_level      = {k: v for k, v in totals.items() if k not in all_children}
        top_total      = sum(top_level.values())
        global_unaccnt = wall_total - top_total
        lines.append(sep)
        lines.append(
            f"{'Untracked ':<{col_w}} {'':>7}  "
            f"{global_unaccnt*scale:>10.3f}  "
            f"{100*global_unaccnt/wall_total:>5.1f}%"
        )

    lines.append(sep)

    output = "\n".join(lines) + "\n"
    if show:
        print(output)

    if path is not None:
        with open(path, "w") as f:
            f.write(output)


def reset():
    """Clear all recorded timings."""
    _timings.clear()
    global _wall_start, _wall_end
    _wall_start = None
    _wall_end   = None
