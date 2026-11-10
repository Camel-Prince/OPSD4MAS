"""GPU filler lifecycle boundaries used only by the FactCard LOO path."""

from contextlib import contextmanager


def _load_gpu_filler_controls():
    from scripts.gpu_filler_control import pause, resume

    return pause, resume


@contextmanager
def fact_card_cpu_filler(reason: str):
    """Fill GPUs during a FactCard CPU/API section, then synchronously pause."""
    try:
        pause, resume = _load_gpu_filler_controls()
    except (ImportError, ModuleNotFoundError):
        yield
        return
    active = resume(reason=reason, wait=True)
    try:
        yield
    finally:
        if active:
            pause(reason=f"after {reason}", wait=True)


@contextmanager
def fact_card_gpu_section(reason: str):
    """Synchronously pause filler for model GPU work, then resume CPU filling."""
    try:
        pause, resume = _load_gpu_filler_controls()
    except (ImportError, ModuleNotFoundError):
        yield
        return
    controlled = pause(reason=reason, wait=True)
    try:
        yield
    finally:
        if controlled:
            resume(reason=f"after {reason}", wait=True)
