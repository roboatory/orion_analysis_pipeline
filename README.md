# Orion

Patch-first Orion multiplex imaging pipeline.

## Commands

```bash
uv run python3 main.py run --configuration configurations/colorectal_cancer_33_01.yaml
```

`run` prints dataset metadata, selects the analysis patch, runs the full patch-first pipeline, and writes outputs under `outputs/<sample_identifier>`.
