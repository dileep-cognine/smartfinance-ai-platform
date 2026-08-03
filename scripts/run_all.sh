#!/usr/bin/env bash
set -euo pipefail
docker compose --profile training run --rm model_optimizer python -m src.main prepare-data
docker compose --profile training run --rm model_optimizer python -m src.main run-all
