#!/usr/bin/env bash
# Run every task notebook in place, one after the other (one training on the GPU at a time).
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
for task in ner relations classification slots decision; do
  rm -rf "demo-$task"
  echo "== $task"
  uv run jupyter execute --inplace "task_$task.ipynb"
done
