#!/usr/bin/env bash
# =============================================================================
# Initialize Neon PostgreSQL with all migrations
# =============================================================================
# Runs all migration files against a Neon (or any PostgreSQL) database.
#
# Usage:
#   export DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/news_tracker?sslmode=require"
#   bash deployments/cloud/neon-init.sh
# =============================================================================
set -euo pipefail

if [ -z "${DATABASE_URL:-}" ]; then
    echo "ERROR: DATABASE_URL environment variable is not set."
    echo "Usage: DATABASE_URL='postgresql://...' bash $0"
    exit 1
fi

MIGRATIONS_DIR="$(cd "$(dirname "$0")/../../migrations" && pwd)"

if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo "ERROR: Migrations directory not found at $MIGRATIONS_DIR"
    exit 1
fi

echo "=== Running migrations against Neon database ==="
echo "    Migrations dir: $MIGRATIONS_DIR"
echo ""

# Run migrations in sorted order
for migration in $(ls "$MIGRATIONS_DIR"/*.sql | sort); do
    filename=$(basename "$migration")
    echo "  Applying: $filename"
    psql "$DATABASE_URL" -f "$migration" -v ON_ERROR_STOP=1 2>&1 | sed 's/^/    /'
    echo "  ✓ $filename applied"
done

echo ""
echo "=== All migrations applied successfully ==="
echo ""
echo "Verify with:"
echo "  psql \"$DATABASE_URL\" -c '\\dt'"
echo "  psql \"$DATABASE_URL\" -c \"SELECT extname FROM pg_extension;\""
