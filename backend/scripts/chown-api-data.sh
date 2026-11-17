#!/usr/bin/env bash
# Give the api's data to the user it runs as (SEG-03 phase 3).
#
# Until v0.106.1 the api ran as root, so everything it wrote (installed
# backends, downloaded models, HF cache, batch files, image outputs) belongs to
# root and the non-root api cannot update or delete it. This changes the owner
# of every folder the api mounts read-write in docker-compose.yml, and only
# those. Run it from the repo root, with the api stopped or right before
# recreating it (files written as root in between need a second run).
#
# Only metadata changes (chown), no data is copied: minutes, even for hundreds
# of GB. Root can still read and write files owned by the new user, so going
# back to running the api as root needs no undo.
#
#   backend/scripts/chown-api-data.sh            # uid/gid from OCABRA_UID/OCABRA_GID or 1000
#   backend/scripts/chown-api-data.sh --dry-run  # only list what it would change
set -euo pipefail

uid="${OCABRA_UID:-1000}"
gid="${OCABRA_GID:-1000}"
dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

mounts="$(docker compose config --format json | python3 -c '
import json, sys
api = json.load(sys.stdin)["services"]["api"]
for v in api.get("volumes", []):
    if v.get("read_only") or v.get("target", "").startswith("/run/"):
        continue
    print(v["type"], v["source"])
')"

args=()
while read -r kind source; do
    [[ -z "${kind}" ]] && continue
    if [[ "${kind}" == "volume" ]]; then
        volume="$(docker compose config --format json | python3 -c '
import json, sys
print(json.load(sys.stdin)["volumes"][sys.argv[1]]["name"])' "${source}")"
        echo "volume ${volume}"
        args+=(-v "${volume}:/chown/${#args[@]}")
    else
        echo "folder ${source}"
        args+=(-v "${source}:/chown/${#args[@]}")
    fi
done <<< "${mounts}"

if (( dry_run )); then
    echo "(dry run: would chown -R ${uid}:${gid} the above)"
    exit 0
fi

echo "chown -R ${uid}:${gid} …"
docker run --rm "${args[@]}" alpine:3.20 chown -R "${uid}:${gid}" /chown
echo "done"
