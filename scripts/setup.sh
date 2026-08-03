#!/usr/bin/env bash
set -euo pipefail
test -f .env || cp .env.example .env
docker compose config --quiet
echo "Configuration created. Replace placeholder secrets in .env before startup."
