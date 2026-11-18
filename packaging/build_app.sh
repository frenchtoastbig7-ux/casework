#!/bin/bash
# Build dist/Casework.app -- a self-contained macOS bundle.
#
#   bash packaging/build_app.sh
#
# Needs a working dev environment (see README) with the `app` extra installed:
#   uv pip install --python .venv -e ".[app]"
# The bundle carries Python and every dependency; the two face models are
# downloaded by the app on first use, not bundled.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HERE/dist/Casework.app"
PY="${PYTHON:-$HERE/.venv/bin/python}"

[ -x "$PY" ] || { echo "no interpreter at $PY -- create one with 'uv venv --python 3.11'"; exit 1; }

echo "==> regenerating icon"
"$PY" "$HERE/packaging/make_icon.py"

echo "==> running PyInstaller (this takes a few minutes)"
rm -rf "$HERE/build" "$HERE/dist"
"$PY" -m PyInstaller "$HERE/packaging/casework-mac.spec" --noconfirm --log-level WARN

[ -d "$APP" ] || { echo "build produced no bundle at $APP"; exit 1; }

# An ad-hoc signature keeps the bundle launchable and stops macOS calling it
# damaged. It is NOT notarised: first launch still needs right-click -> Open.
echo "==> ad-hoc signing"
codesign --force --deep --sign - "$APP" 2>/dev/null || \
  echo "    (codesign unavailable -- bundle still runs after right-click -> Open)"

xattr -cr "$APP" 2>/dev/null || true

echo "==> done: $APP  ($(du -sh "$APP" | cut -f1))"
