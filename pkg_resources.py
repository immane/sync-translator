try:
    # Prefer importlib.metadata on modern Python
    from importlib.metadata import distribution as _distribution

    class _Dist:
        def __init__(self, name):
            try:
                self.version = _distribution(name).version
            except Exception:
                self.version = "0"

    def get_distribution(name):
        return _Dist(name)

except Exception:
    # Minimal fallback if importlib.metadata not available
    class _Dist:
        version = "0"

    def get_distribution(name):
        return _Dist()
