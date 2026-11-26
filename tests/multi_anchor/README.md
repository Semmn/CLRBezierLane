# Multi-anchor head tests (CPU)

Stubs for mmcv / mmdet / mmengine are in `stubs/` (the tests do not need them installed).

```
cd tests/multi_anchor
python test_multi_anchor.py ../.. ORIG_TREE        # ORIG_TREE: a checkout of main before the change (switch-off check)
python test_multi_anchor_overfit.py ../.. 400      # ~10 min on 4 CPU threads
python cost_multi_anchor.py ../..                  # GMACs / CPU ms / params vs the pointcond pair
```

Outputs: `*_output.txt`, `cost_multi_anchor_output.json`.
