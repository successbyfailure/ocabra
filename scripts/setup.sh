#!/usr/bin/env bash
set -euo pipefail

echo "=== oCabra Setup ==="

# 1. Copy .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    # Generate a random SECRET_KEY
    if command -v openssl &>/dev/null; then
        SECRET=$(openssl rand -hex 32)
        sed -i "s/^SECRET_KEY=.*/SECRET_KEY=${SECRET}/" .env
        echo "✓ Generated random SECRET_KEY"
    fi
    echo "✓ Created .env from .env.example — edit it before proceeding."
    echo "  Minimum required: POSTGRES_PASSWORD, DATABASE_URL"
    exit 0
fi

# 2. Create data directories
mkdir -p data/models data/hf_cache
echo "✓ Data directories ready"

# 3. Start dependencies only
docker compose up -d postgres redis
echo "✓ Waiting for postgres and redis..."

# Wait for postgres to be ready (up to 30s)
for i in $(seq 1 30); do
    if docker compose exec postgres pg_isready -U ocabra -q 2>/dev/null; then
        break
    fi
    sleep 1
done
echo "✓ Postgres is ready"

# 4. Run migrations
docker compose run --rm api alembic upgrade head
echo "✓ Database migrations applied"

# 5. Start all services
docker compose up -d
echo "✓ All services started"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  oCabra is running!"
echo "  Web UI:   http://localhost"
echo "  API docs: http://localhost/docs"
echo "  Health:   http://localhost/health"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
