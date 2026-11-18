#!/bin/sh
# Continuous deployment on the VM, run every minute by handbook-autodeploy.timer (as root).
#
# 1. bring the checkout to the latest main (compose file, Caddyfile; .env is never in git)
# 2. pull the app image CI pushed to Artifact Registry
# 3. if anything changed, restart the stack and wait for /api/health
# 4. if the new version is not healthy within ~90 s, roll back to the image that was running
#
# Nothing reaches into the server from outside: GitHub only pushes the image, the VM pulls it.
set -eu
APP_DIR="${APP_DIR:-/home/Rafay-Dev/handbook-rag-assistant}"
cd "$APP_DIR"
git() { command git -c safe.directory="$APP_DIR" "$@"; }
compose() { docker compose --profile https "$@"; }

before_commit="$(git rev-parse HEAD)"
git fetch --quiet origin main
git reset --quiet --hard origin/main  # tracked files only; .env and other untracked files stay
after_commit="$(git rev-parse HEAD)"

grep -q 'WEB_IMAGE' docker-compose.yml || exit 0  # a checkout from before the registry setup: nothing to do

# keep this script itself up to date: a newer version from GitHub takes over from the next run
installed=/usr/local/bin/handbook-autodeploy
if [ -f "$installed" ] && ! cmp -s deploy/vm/autodeploy.sh "$installed"; then
  install -m 755 deploy/vm/autodeploy.sh "$installed" && echo "updated the deploy script for the next run"
fi

running_image="$(docker inspect --format '{{.Image}}' "$(compose ps -q web 2>/dev/null)" 2>/dev/null || true)"
compose pull --quiet web
# the web service's own image (`config --images web` also lists Chroma, in no fixed order)
image_name="$(compose config --format json | python3 -c 'import json, sys; print(json.load(sys.stdin)["services"]["web"]["image"])')"
pulled_image="$(docker image inspect --format '{{.Id}}' "$image_name")"

if [ "$pulled_image" = "$running_image" ] && [ "$before_commit" = "$after_commit" ]; then
  exit 0  # nothing new
fi

echo "deploying commit ${after_commit%"${after_commit#???????}"} image ${pulled_image#sha256:}"
compose up -d --no-build --remove-orphans

healthy() {
  i=0
  while [ $i -lt 30 ]; do
    curl -fsS --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1 && return 0
    i=$((i + 1)); sleep 3
  done
  return 1
}

if healthy; then
  echo "healthy: $(curl -fsS http://127.0.0.1:8000/api/health | grep -o '"version": "[^"]*"')"
  docker image prune -f >/dev/null 2>&1 || true
  exit 0
fi

echo "new version is not healthy; rolling back" >&2
compose logs --tail 30 web >&2 || true
if [ -n "$running_image" ]; then
  docker tag "$running_image" "$image_name"
  compose up -d --no-build web
  healthy && echo "rolled back to ${running_image#sha256:}" >&2
fi
exit 1
