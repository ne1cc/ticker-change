#!/usr/bin/env bash
# Pull-based deploy for the Lightsail host (run by ticker-change-deploy.timer).
# Deploys the newest main commit whose Tests workflow passed, swapping the
# container only after the new one answers /health; otherwise the previous
# container is restored and the commit is skipped until main moves on.
set -euo pipefail

repo=ne1cc/ticker-change
root=/srv/ticker-change
failed=$root/failed-sha

sha=$(curl -fsS --max-time 20 \
  "https://api.github.com/repos/$repo/actions/workflows/tests.yml/runs?branch=main&event=push&status=success&per_page=1" |
  python3 -c 'import json, sys; runs = json.load(sys.stdin)["workflow_runs"]; print(runs[0]["head_sha"] if runs else "")')
image=ticker-change:$sha
prev_image=$(docker inspect ticker-change --format '{{.Config.Image}}')

[ -n "$sha" ] || exit 0
[ "$image" != "$prev_image" ] || exit 0
[ "$sha" != "$(cat "$failed" 2>/dev/null)" ] || exit 0

echo "deploying $sha (replacing $prev_image)"
echo "$sha" > "$failed"
release=$root/releases/$sha
mkdir -p "$release"
curl -fsSL --max-time 120 "https://codeload.github.com/$repo/tar.gz/$sha" |
  tar --no-same-owner --strip-components=1 -xz -C "$release"
docker build -q -t "$image" "$release"

healthy() {
  for _ in $(seq 1 40); do
    curl -fsS --max-time 5 http://127.0.0.1:8000/health && return 0
    sleep 3
  done
  return 1
}

docker rm -f ticker-change-prev >/dev/null 2>&1 || true
docker stop ticker-change >/dev/null
docker rename ticker-change ticker-change-prev
if docker run -d --name ticker-change --restart unless-stopped \
     --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges:true \
     --log-opt max-size=10m --log-opt max-file=3 \
     -p 127.0.0.1:8000:8000 -v "$root/data:/data" \
     -e DB_PATH=/data/stocks.db -e MODEL_PATH=/data/model.pkl -e HOME=/tmp \
     -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
     -e ALPACA_API_KEY_ID -e ALPACA_API_SECRET_KEY -e GEMINI_API_KEY -e FINNHUB_API_KEY -e SETTINGS_PASSWORD \
     -e ANTHROPIC_MODEL -e OPENAI_MODEL -e GEMINI_MODEL -e OPENROUTER_MODEL \
     "$image" >/dev/null && healthy; then
  docker rm ticker-change-prev >/dev/null
  rm -f "$failed"
else
  echo "health check failed; restoring $prev_image" >&2
  docker logs --tail 100 ticker-change || true
  docker rm -f ticker-change >/dev/null || true
  docker rename ticker-change-prev ticker-change
  docker start ticker-change >/dev/null
  exit 1
fi

# Keep only the running release and the one it replaced.
for img in $(docker images ticker-change --format '{{.Repository}}:{{.Tag}}'); do
  [ "$img" = "$image" ] || [ "$img" = "$prev_image" ] || docker rmi "$img" >/dev/null || true
done
find "$root/releases" -mindepth 1 -maxdepth 1 ! -name "$sha" ! -name "${prev_image#ticker-change:}" -exec rm -rf {} +
echo "deployed $sha"
