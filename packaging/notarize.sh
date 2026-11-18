#!/bin/bash
# Sign, notarise and staple the app + dmg with a Developer ID.
#
#   bash packaging/notarize.sh
#
# Prerequisites (each is yours to do once -- see README "Notarising"):
#   1. A "Developer ID Application" certificate + its private key in the login
#      keychain. Check with:  security find-identity -v -p codesigning
#   2. A stored notarytool credential profile. Create it once with YOUR
#      Apple ID and an app-specific password (this is the step that handles
#      your secret, so you run it, not the build):
#        xcrun notarytool store-credentials casework-notary \
#          --apple-id "you@example.com" --team-id "TEAMID" \
#          --password "app-specific-password"
#
# Override the identity or profile via env if auto-detection is not what you
# want:  IDENTITY="Developer ID Application: Name (TEAMID)" NOTARY_PROFILE=name
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HERE/dist/Casework.app"
DMG="$HERE/dist/Casework.dmg"
ENTITLEMENTS="$HERE/packaging/entitlements.plist"
PROFILE="${NOTARY_PROFILE:-casework-notary}"

[ -d "$APP" ] || { echo "no app at $APP -- run build_app.sh first"; exit 1; }

# --- resolve the signing identity ----------------------------------------
if [ -z "${IDENTITY:-}" ]; then
  # macOS ships bash 3.2, which has no `mapfile`; read into the array by hand.
  IDS=()
  while IFS= read -r line; do
    [ -n "$line" ] && IDS+=("$line")
  done < <(security find-identity -v -p codesigning 2>/dev/null \
           | grep "Developer ID Application" | sed -E 's/.*"(.*)"/\1/')
  case "${#IDS[@]}" in
    0) echo "ERROR: no 'Developer ID Application' identity in the keychain."
       echo "Install your Developer ID certificate first (see README)."; exit 1 ;;
    1) IDENTITY="${IDS[0]}" ;;
    *) echo "Multiple Developer ID identities found; set IDENTITY to one of:"
       printf '  %s\n' "${IDS[@]}"; exit 1 ;;
  esac
fi
echo "==> signing identity: $IDENTITY"

if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
  echo "ERROR: no usable notarytool profile '$PROFILE'."
  echo "Create it once (this handles your Apple credentials, so you run it):"
  echo "  xcrun notarytool store-credentials $PROFILE \\"
  echo "    --apple-id \"you@example.com\" --team-id \"TEAMID\" \\"
  echo "    --password \"app-specific-password\""
  exit 1
fi

# --- sign inside-out ------------------------------------------------------
# Every nested Mach-O must carry the hardened runtime and a secure timestamp
# before the outer bundle is sealed. Signing inside-out rather than with
# --deep is what makes a large PyInstaller tree notarise reliably: --deep is
# documented as not for this, and skips exactly the odd nested dylib that then
# fails notarisation with an opaque error.
echo "==> signing nested libraries (this takes a minute)"
while IFS= read -r -d '' f; do
  codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$f" 2>/dev/null || {
      echo "    failed to sign $f"; exit 1; }
done < <(find "$APP/Contents" \( -name "*.dylib" -o -name "*.so" \) -print0)

echo "==> signing the app bundle"
codesign --force --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

# --- notarise the app (zip is the submission container) -------------------
ZIP="$HERE/dist/Casework-notarize.zip"
rm -f "$ZIP"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"
echo "==> submitting app to Apple (waits for the verdict)"
xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
rm -f "$ZIP"

echo "==> stapling the app"
xcrun stapler staple "$APP"

# --- build the dmg from the stapled app, then notarise the dmg too --------
# The dmg contains an already-stapled app, so a copy dragged out validates
# offline; notarising and stapling the dmg itself makes the downloaded image
# pass Gatekeeper without a network round-trip on first mount.
echo "==> building dmg from the stapled app"
bash "$HERE/packaging/make_dmg.sh" "$APP" "$DMG"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
echo "==> submitting dmg to Apple"
xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$DMG"

# --- verify as Gatekeeper will ------------------------------------------
echo "==> verification"
echo "-- app --";  spctl -a -vvv --type execute "$APP"    2>&1 | sed 's/^/   /'
echo "-- app --";  xcrun stapler validate "$APP"          2>&1 | sed 's/^/   /'
echo "-- dmg --";  xcrun stapler validate "$DMG"          2>&1 | sed 's/^/   /'
echo
echo "==> done: $DMG is signed, notarised and stapled."
echo "    It now opens with a normal double-click on any Mac."
