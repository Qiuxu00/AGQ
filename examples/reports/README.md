# Generated matrix comparison reports

These two reports were produced by the actual repository commands on 2026-10-12. They contain scalar metrics from fixed-seed generated matrices, with no pretrained weights or robot data.

```bash
agq compare --methods rtn,hqq --bits 4 --output outputs/matrix_w4.json
agq compare --methods rtn,hqq,quanto --bits 8 --output outputs/matrix_w8.json
```

See [the baseline guide](../../docs/BASELINES.md) for settings, tested versions, measured quantities, and interpretation. They are examples of the comparison interface, not additional paper results.
