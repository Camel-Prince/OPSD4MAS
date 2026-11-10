"""Compatibility hooks for historical imports; the public release has no GPU filler."""

def pause(*args, **kwargs):
    return False

def resume(*args, **kwargs):
    return False
