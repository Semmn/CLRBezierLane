class _R:
    """Recording registry: build(dict(type=...)) looks the class up by name (an "mmdet." prefix is ignored)."""
    def __init__(self): self._m = {}
    def register_module(self, *a, **k):
        def deco(c):
            self._m[k.get("name") or c.__name__] = c; return c
        if a and isinstance(a[0], type):
            return deco(a[0])
        return deco
    def get(self, name): return self._m.get(name.split(".")[-1]) if name else None
    def build(self, cfg):
        cfg = dict(cfg); t = cfg.pop("type"); return (self.get(t) if isinstance(t, str) else t)(**cfg)
TRANSFORMS = _R(); MODELS = _R(); DATASETS = _R(); METRICS = _R()
