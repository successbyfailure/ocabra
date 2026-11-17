#!/usr/bin/env bash
# Publish an oCabra release from this machine (what .github/workflows/release.yml
# does when there are Actions minutes):
#
#   backend/scripts/publish-release.sh 0.98.0            # api + frontend + node
#   backend/scripts/publish-release.sh 0.98.0 --no-node  # skip the CUDA node image
#
# 1. The tag vX.Y.Z must exist (git tag vX.Y.Z && git push origin vX.Y.Z) and
#    match APP_VERSION and frontend/package.json at that commit.
# 2. Builds from a clean worktree of the tag (never the working tree: other
#    sessions' uncommitted work must not end up in a release).
# 3. Pushes ghcr.io/<owner>/ocabra-{api,frontend,node}:X.Y.Z and :latest with
#    `gh auth token` (needs write:packages), then creates/updates the GitHub
#    release with that version's changelog section.
#
# Publishing is what makes nodes with policy "auto" update themselves: only
# run it when the release is meant to go out.
set -euo pipefail

version="${1:?usage: publish-release.sh X.Y.Z [--no-node]}"
version="${version#v}"
build_node=1
[ "${2:-}" = "--no-node" ] && build_node=0
owner="${OCABRA_RELEASE_OWNER:-successbyfailure}"
registry="ghcr.io/${owner}"
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
tag="v${version}"

git -C "$repo_root" fetch --quiet origin tag "$tag" 2>/dev/null || true
git -C "$repo_root" rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null \
    || { echo "tag ${tag} does not exist (git tag ${tag} && git push origin ${tag})" >&2; exit 1; }

wt="$(mktemp -d /tmp/ocabra-release-XXXXXX)"
cleanup() { git -C "$repo_root" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"; }
trap cleanup EXIT
git -C "$repo_root" worktree add --quiet --detach "$wt" "$tag"

app=$(sed -n 's/^APP_VERSION = "\(.*\)"/\1/p' "$wt/backend/ocabra/config.py")
front=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$wt/frontend/package.json")
if [ "$app" != "$version" ] || [ "$front" != "$version" ]; then
    echo "tag ${version}, APP_VERSION ${app}, package.json ${front}: they must match" >&2
    exit 1
fi

gh auth token | docker login ghcr.io -u "${owner}" --password-stdin >/dev/null

publish() {  # name context dockerfile
    local name=$1 context=$2 file=$3
    echo "=== ${name}:${version}"
    docker build -q -f "$file" \
        --label "org.opencontainers.image.source=https://github.com/${owner}/ocabra" \
        --label "org.opencontainers.image.version=${version}" \
        -t "${registry}/${name}:${version}" -t "${registry}/${name}:latest" "$context"
    docker push -q "${registry}/${name}:${version}"
    docker push -q "${registry}/${name}:latest"
}

publish ocabra-api "$wt/backend" "$wt/backend/Dockerfile.slim"
publish ocabra-frontend "$wt/frontend" "$wt/frontend/Dockerfile"
[ "$build_node" = 1 ] && publish ocabra-node "$wt/backend" "$wt/backend/Dockerfile.node"

notes="$(mktemp)"
awk -v v="$version" '
  $0 ~ "^## " v " " {on=1; next}
  on && /^---$/ {exit}
  on {print}
' "$wt/backend/ocabra/docs_content/changelog.md" > "$notes"
[ -s "$notes" ] || echo "Ver /docs/changelog." > "$notes"
if gh release view "$tag" --repo "${owner}/ocabra" >/dev/null 2>&1; then
    gh release edit "$tag" --repo "${owner}/ocabra" --notes-file "$notes"
else
    gh release create "$tag" --repo "${owner}/ocabra" --title "oCabra ${version}" --notes-file "$notes"
fi
rm -f "$notes"
echo "released ${version}"
