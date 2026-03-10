#!/usr/bin/env bash
set -euo pipefail

echo "=== oCabra Setup ==="

# 1. Copy .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example — edit it before proceeding."
    echo "  Minimum required: POSTGRES_PASSWORD, DATABASE_URL"
    exit 0
fi

# 2. Create data directories
mkdir -p data/models data/hf_cache
echo "✓ Data directories ready"

# 3. Start dependencies
docker compose up -d postgres redis
echo "✓ Waiting for postgres and redis..."
sleep 5

# 4. Run migrations
docker compose run --rm api alembic upgrade head
echo "✓ Database migrations applied"

# 5. Start all services
docker compose up -d
echo "✓ All services started"

echo ""
echo "oCabra is running at http://localhost"
echo "API docs at http://localhost/docs"
