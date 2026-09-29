default:
    just --list

sync:
    uv sync

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

test:
    uv run pytest -q

check: lint fmt-check test

run-smoke:
    HF_HUB_OFFLINE=1 uv run pytest -q tests/test_run_smoke.py

# Keep the busy GPU and live teacher servers out of verification.
check-cpu: lint fmt-check
    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 uv run pytest -q -m 'not model and not teacher'

# Run manually when the shared GPU is available. Each output directory must be new.
relation-memory head out steps="400" lr="3e-4" neg_ratio="1":
    HF_HUB_OFFLINE=1 uv run python scripts/relation_memory_test.py --relation-head {{head}} --steps {{steps}} --lr {{lr}} --neg-ratio {{neg_ratio}} --out {{out}}

# Teacher labelling time for the cost figures. Needs a free GPU and the teacher server up.
time-teacher teacher dataset locale="en-US":
    HF_HUB_OFFLINE=1 uv run active-gliner time-teacher --teacher {{teacher}} --dataset {{dataset}} --locale {{locale}}

# Zero-shot reference ladder (lower bound): inference only. Finished pairs are skipped.
zero-shot models="":
    HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run active-gliner zero-shot {{ if models == "" { "" } else { "--models " + models } }}
