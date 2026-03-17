# Orion

Patch-first Orion multiplex imaging pipeline.

## Commands

```bash
uv run python3 main.py run --configuration configurations/CRC33_01.yaml --mode patch
uv run python3 main.py run --configuration configurations/CRC33_01.yaml --mode whole-slide
```

`patch` is the implemented mode. `whole-slide` is a reserved development flag that currently exits with a clear not-implemented message.
