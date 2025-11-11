#!/usr/bin/env bash
set -euo pipefail

for plot in plots/*.py; do
  [ "$(basename "$plot")" = "__init__.py" ] && continue
  echo "Running $plot"
  uv run "$plot"
done
