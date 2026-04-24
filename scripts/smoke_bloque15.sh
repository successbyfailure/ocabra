#!/usr/bin/env bash
# Smoke test for Bloque 15 (Backends Modulares) — Fase 1 endpoints.
# Exercises /ocabra/backends from inside the api container, exits non-zero on
# any assertion failure.  Requires docker compose stack to be running.

set -euo pipefail

user="${OCABRA_ADMIN_USER:-ocabra}"
password="${OCABRA_ADMIN_PASSWORD:-ocabra}"

docker compose exec -T api bash -c "
set -euo pipefail

curl -sf -c /tmp/c.txt -X POST http://localhost:8000/ocabra/auth/login \\
  -H 'Content-Type: application/json' \\
  -d '{\"username\":\"$user\",\"password\":\"$password\"}' > /dev/null

# 1. List returns 12 backends, all built-in until Fase 2 migrates them.
count=\$(curl -sf -b /tmp/c.txt http://localhost:8000/ocabra/backends \\
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d))')
[ \"\$count\" -eq 12 ] || { echo \"expected 12 backends, got \$count\"; exit 1; }

# 2. Single backend detail works.
curl -sf -b /tmp/c.txt http://localhost:8000/ocabra/backends/ollama > /dev/null

# 3. Unknown backend returns 404.
rc=\$(curl -s -o /dev/null -w '%{http_code}' -b /tmp/c.txt \\
  http://localhost:8000/ocabra/backends/nope)
[ \"\$rc\" = '404' ] || { echo \"expected 404 for nope, got \$rc\"; exit 1; }

# 4. Uninstalling a built-in is rejected with 409.
rc=\$(curl -s -o /dev/null -w '%{http_code}' -X POST -b /tmp/c.txt \\
  http://localhost:8000/ocabra/backends/whisper/uninstall)
[ \"\$rc\" = '409' ] || { echo \"expected 409 for built-in uninstall, got \$rc\"; exit 1; }

# 5. OCI install is still 501 until Fase 3 lands.
rc=\$(curl -s -o /dev/null -w '%{http_code}' -X POST -b /tmp/c.txt \\
  -H 'Content-Type: application/json' -d '{\"method\":\"oci\"}' \\
  http://localhost:8000/ocabra/backends/whisper/install)
[ \"\$rc\" = '501' ] || { echo \"expected 501 for oci install, got \$rc\"; exit 1; }

echo 'smoke_bloque15: 5/5 OK'
"
