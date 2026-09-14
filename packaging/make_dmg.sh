#!/bin/bash
# Build a compressed drag-to-Applications .dmg from the built app bundle.
#
#   bash packaging/make_dmg.sh [dist/Casework.app] [out.dmg]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HERE/dist/Casework.app}"
OUT="${2:-$HERE/dist/Casework.dmg}"
VOL="Casework"

[ -d "$APP" ] || { echo "no app bundle at $APP -- run build_app.sh first"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"      # drag target

cat > "$STAGE/READ ME FIRST.txt" <<'NOTE'
Casework
========

Scoped subject collection and face-clustering workbench, for investigators
working under a defined mandate. Everything runs on this Mac; nothing is
uploaded, and there is no account.

Install
  Drag "Casework" onto the Applications folder shown here.

First launch
  macOS blocks apps from unidentified developers. The first time only:
  right-click (or Control-click) the app in Applications and choose "Open",
  then confirm. Afterwards it opens normally by double-clicking.

Face models
  The first time you run face detection, the app downloads two models
  (about 39 MB total) into ~/.casework/models. After that everything works
  offline. Collecting and reviewing material needs no download.

Your data
  Cases, collected material, the audit log and the models all live in
  ~/.casework  -- one folder per case under ~/.casework/cases. The command
  line tool (if installed) and this app share that folder. Delete it to
  remove everything the app has stored.

Scope
  Casework collects only from sources you attribute to a subject you have
  already identified, and matches only against reference images you enrol.
  It builds no index of the general public and has no "who is this?" search.
NOTE

rm -f "$OUT"
mkdir -p "$(dirname "$OUT")"
hdiutil create -volname "$VOL" -srcfolder "$STAGE" -fs HFS+ \
  -format UDZO -imagekey zlib-level=9 -ov "$OUT" >/dev/null

echo "wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
