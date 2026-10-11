#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT/Voxt.xcodeproj"
SCHEME="Voxt"
CONFIGURATION="Release"
NATIVE_ARCH="$(uname -m)"
DESTINATION="platform=macOS,arch=$NATIVE_ARCH"
DERIVED_DATA="$ROOT/build/LocalPackageDerivedData"
OUTPUT_ROOT="$ROOT/build/local-package"
INSTALL_DIR="/Applications"
INSTALL_APP=false
OPEN_AFTER_INSTALL=false
CLEAN_BUILD=true
LOCAL_OPTIMIZATION=true
UNIVERSAL_BUILD=false
SIGN_APP=true
APP_SIGN_IDENTITY="${DEVELOPER_ID_APP_IDENTITY:-}"
INSTALLER_SIGN_IDENTITY="${DEVELOPER_ID_INSTALLER_IDENTITY:-}"
PROVISIONING_PROFILE=""
NOTARY_KEY=""
NOTARY_KEY_ID=""
NOTARY_ISSUER=""
VERSION=""
BUILD_NUMBER=""
APP_NAME="Voxt"
NATIVE_RUNTIME_DIR="${VOXT_NATIVE_RUNTIME_DIR:-}"

usage() {
  cat <<'EOF'
Usage: tools/package_local_app.sh [options]

Builds a local Release app and packages it for manual verification. This does
not create a GitHub release and does not publish Sparkle appcast metadata.

Options:
  --version VERSION        Override MARKETING_VERSION for this local build.
  --build-number NUMBER   Override CURRENT_PROJECT_VERSION. Defaults to a timestamp.
  --output-dir PATH       Output directory. Defaults to build/local-package.
  --derived-data PATH     DerivedData directory. Defaults to build/LocalPackageDerivedData.
  --install               Install the built app into /Applications after packaging.
  --install-dir PATH      Install target directory. Defaults to /Applications.
  --open                  Open the app after --install.
  --no-clean              Do not run xcodebuild clean before build.
  --optimized             Use Release optimization. Default disables Swift optimization
                          for local packaging to avoid local optimizer crashes.
  --universal             Build both arm64 and x86_64 via generic macOS destination.
  --app-sign-identity ID  Developer ID Application identity for app signing.
  --pkg-sign-identity ID  Developer ID Installer identity for pkg signing.
  --provisioning-profile PATH
                          Embed this Developer ID provisioning profile. Signing keeps
                          the Keychain entitlements only with a profile.
  --notary-key PATH       App Store Connect API key (.p8) for notarization. With
                          --notary-key-id and --notary-issuer, the app is notarized
                          and stapled before packaging, and the DMG and PKG after.
  --notary-key-id ID      App Store Connect API key ID.
  --notary-issuer ID      App Store Connect API issuer ID.
  --native-runtime PATH   Embed a built Omni runtime in the app bundle.
  --no-sign               Do not sign the app after building.
  -h, --help              Show this help.

Optional environment:
  DEVELOPER_ID_APP_IDENTITY        Signs the app when set.
  DEVELOPER_ID_INSTALLER_IDENTITY  Signs the .pkg when set.
  VOXT_NATIVE_RUNTIME_DIR          Runtime directory used when --native-runtime is omitted.

Examples:
  tools/package_local_app.sh
  tools/package_local_app.sh --install --open
  tools/package_local_app.sh --version 1.12.0 --build-number 101200099 --install
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo
  echo "==> $*"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      [[ -n "$VERSION" ]] || die "--version requires a value"
      shift 2
      ;;
    --build-number)
      BUILD_NUMBER="${2:-}"
      [[ -n "$BUILD_NUMBER" ]] || die "--build-number requires a value"
      shift 2
      ;;
    --output-dir)
      OUTPUT_ROOT="${2:-}"
      [[ -n "$OUTPUT_ROOT" ]] || die "--output-dir requires a value"
      shift 2
      ;;
    --derived-data)
      DERIVED_DATA="${2:-}"
      [[ -n "$DERIVED_DATA" ]] || die "--derived-data requires a value"
      shift 2
      ;;
    --install)
      INSTALL_APP=true
      shift
      ;;
    --install-dir)
      INSTALL_DIR="${2:-}"
      [[ -n "$INSTALL_DIR" ]] || die "--install-dir requires a value"
      shift 2
      ;;
    --open)
      OPEN_AFTER_INSTALL=true
      INSTALL_APP=true
      shift
      ;;
    --no-clean)
      CLEAN_BUILD=false
      shift
      ;;
    --optimized)
      LOCAL_OPTIMIZATION=false
      shift
      ;;
    --universal)
      UNIVERSAL_BUILD=true
      DESTINATION="generic/platform=macOS"
      shift
      ;;
    --app-sign-identity)
      APP_SIGN_IDENTITY="${2:-}"
      [[ -n "$APP_SIGN_IDENTITY" ]] || die "--app-sign-identity requires a value"
      shift 2
      ;;
    --pkg-sign-identity)
      INSTALLER_SIGN_IDENTITY="${2:-}"
      [[ -n "$INSTALLER_SIGN_IDENTITY" ]] || die "--pkg-sign-identity requires a value"
      shift 2
      ;;
    --provisioning-profile)
      PROVISIONING_PROFILE="${2:-}"
      [[ -n "$PROVISIONING_PROFILE" ]] || die "--provisioning-profile requires a value"
      shift 2
      ;;
    --notary-key)
      NOTARY_KEY="${2:-}"
      [[ -n "$NOTARY_KEY" ]] || die "--notary-key requires a value"
      shift 2
      ;;
    --notary-key-id)
      NOTARY_KEY_ID="${2:-}"
      [[ -n "$NOTARY_KEY_ID" ]] || die "--notary-key-id requires a value"
      shift 2
      ;;
    --notary-issuer)
      NOTARY_ISSUER="${2:-}"
      [[ -n "$NOTARY_ISSUER" ]] || die "--notary-issuer requires a value"
      shift 2
      ;;
    --native-runtime)
      NATIVE_RUNTIME_DIR="${2:-}"
      [[ -n "$NATIVE_RUNTIME_DIR" ]] || die "--native-runtime requires a value"
      shift 2
      ;;
    --no-sign)
      SIGN_APP=false
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -d "$PROJECT" ]] || die "Xcode project not found: $PROJECT"
command -v xcodebuild >/dev/null || die "xcodebuild is required"
command -v hdiutil >/dev/null || die "hdiutil is required"
command -v productbuild >/dev/null || die "productbuild is required"

if [[ -n "$NATIVE_RUNTIME_DIR" ]]; then
  [[ -d "$NATIVE_RUNTIME_DIR/bin" ]] || die "native runtime bin directory not found: $NATIVE_RUNTIME_DIR/bin"
  for required_runtime_file in \
    qwen3_asr_server \
    whisper_server \
    cohere_transcribe_server \
    moss_transcribe_diarize_server \
    nemotron_asr_server \
    parakeet_server \
    sensevoice_server \
    libmlx.dylib \
    libjaccl.dylib \
    mlx.metallib; do
    [[ -f "$NATIVE_RUNTIME_DIR/bin/$required_runtime_file" ]] || die "native runtime file not found: $NATIVE_RUNTIME_DIR/bin/$required_runtime_file"
  done
fi

if [[ "$OPEN_AFTER_INSTALL" == "true" && "$INSTALL_APP" != "true" ]]; then
  die "--open requires --install"
fi

if [[ -n "$PROVISIONING_PROFILE" ]]; then
  [[ -f "$PROVISIONING_PROFILE" ]] || die "provisioning profile not found: $PROVISIONING_PROFILE"
  [[ "$SIGN_APP" == "true" ]] || die "--provisioning-profile cannot be combined with --no-sign"
fi

NOTARIZE=false
if [[ -n "$NOTARY_KEY$NOTARY_KEY_ID$NOTARY_ISSUER" ]]; then
  [[ -n "$NOTARY_KEY" && -n "$NOTARY_KEY_ID" && -n "$NOTARY_ISSUER" ]] \
    || die "--notary-key, --notary-key-id and --notary-issuer go together"
  [[ -f "$NOTARY_KEY" ]] || die "notary key not found: $NOTARY_KEY"
  [[ "$SIGN_APP" == "true" ]] || die "notarization cannot be combined with --no-sign"
  NOTARIZE=true
fi

notarize() {
  local artifact="$1"
  local result_plist submission_id status
  result_plist="$(mktemp)"
  log "Notarizing $(basename "$artifact")"
  xcrun notarytool submit "$artifact" --key "$NOTARY_KEY" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER" \
    --wait --timeout 60m --output-format plist > "$result_plist" || true
  submission_id="$(plutil -extract id raw -o - "$result_plist" 2>/dev/null || true)"
  status="$(plutil -extract status raw -o - "$result_plist" 2>/dev/null || true)"
  rm -f "$result_plist"
  [[ -n "$submission_id" ]] || die "notarytool did not accept $artifact for notarization"
  echo "$(basename "$artifact"): $status ($submission_id)"
  if [[ "$status" != "Accepted" ]]; then
    xcrun notarytool log "$submission_id" --key "$NOTARY_KEY" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER" || true
    die "notarization of $artifact ended with status '$status'"
  else
    :
  fi
}

if [[ -z "$VERSION" ]]; then
  VERSION="$(
    xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration "$CONFIGURATION" -showBuildSettings \
      | awk -F' = ' '/ MARKETING_VERSION = / {print $2; exit}'
  )"
fi
[[ -n "$VERSION" ]] || die "unable to resolve MARKETING_VERSION"

if [[ -z "$BUILD_NUMBER" ]]; then
  BUILD_NUMBER="$(date +%Y%m%d%H%M)"
fi
[[ "$BUILD_NUMBER" =~ ^[0-9]+$ ]] || die "--build-number must be numeric"

if [[ "$SIGN_APP" == "true" && -z "$APP_SIGN_IDENTITY" ]]; then
  APP_SIGN_IDENTITY="$(
    security find-identity -v -p codesigning 2>/dev/null \
      | awk '/Developer ID Application/ {print $2; exit}'
  )"
fi

if [[ -z "$INSTALLER_SIGN_IDENTITY" ]]; then
  INSTALLER_SIGN_IDENTITY="$(
    security find-identity -v -p basic 2>/dev/null \
      | awk -F '"' '/Developer ID Installer/ {print $2; exit}'
  )"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/$VERSION-$BUILD_NUMBER-$STAMP"
APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/$APP_NAME.app"
ZIP_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION-macOS.zip"
DMG_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION.dmg"
PKG_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION.pkg"
DMG_STAGING_DIR="$OUTPUT_DIR/dmg-staging"

mkdir -p "$OUTPUT_DIR"

log "Building $APP_NAME $VERSION ($BUILD_NUMBER)"
BUILD_ACTIONS=()
if [[ "$CLEAN_BUILD" == "true" ]]; then
  BUILD_ACTIONS+=(clean)
fi
BUILD_ACTIONS+=(build)

BUILD_OVERRIDES=(
  MARKETING_VERSION="$VERSION"
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER"
  ENABLE_DEBUG_DYLIB=NO
  CODE_SIGNING_ALLOWED=NO
  CODE_SIGNING_REQUIRED=NO
  CODE_SIGN_IDENTITY=
)
if [[ "$UNIVERSAL_BUILD" != "true" ]]; then
  BUILD_OVERRIDES+=(ONLY_ACTIVE_ARCH=YES)
fi
if [[ "$LOCAL_OPTIMIZATION" == "true" ]]; then
  BUILD_OVERRIDES+=(
    SWIFT_OPTIMIZATION_LEVEL=-Onone
    GCC_OPTIMIZATION_LEVEL=0
  )
fi

xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "$DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  -skipPackagePluginValidation \
  "${BUILD_ACTIONS[@]}" \
  "${BUILD_OVERRIDES[@]}"

[[ -d "$APP_PATH" ]] || die "built app not found: $APP_PATH"

if [[ -n "$NATIVE_RUNTIME_DIR" ]]; then
  log "Embedding Omni native runtime"
  OMNI_RUNTIME_DESTINATION="$APP_PATH/Contents/Resources/OmniRuntime"
  rm -rf "$OMNI_RUNTIME_DESTINATION"
  # The native build output also contains .build/ (CMake files and a local
  # Python virtual environment). It is build machinery, not part of the
  # runtime, and may contain symlinks into the developer's machine that make
  # the signed app invalid. Embed only the installed runtime binaries and
  # resources from bin/.
  mkdir -p "$OMNI_RUNTIME_DESTINATION"
  ditto "$NATIVE_RUNTIME_DIR/bin" "$OMNI_RUNTIME_DESTINATION/bin"
fi

log "Validating app bundle"
PLIST_PATH="$APP_PATH/Contents/Info.plist"
SHORT_VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$PLIST_PATH" 2>/dev/null || true)"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleVersion" "$PLIST_PATH" 2>/dev/null || true)"
[[ "$SHORT_VERSION" == "$VERSION" ]] || die "CFBundleShortVersionString is $SHORT_VERSION, expected $VERSION"
[[ "$BUNDLE_VERSION" == "$BUILD_NUMBER" ]] || die "CFBundleVersion is $BUNDLE_VERSION, expected $BUILD_NUMBER"

if [[ "$SIGN_APP" == "true" && -n "$APP_SIGN_IDENTITY" ]]; then
  log "Signing app with $APP_SIGN_IDENTITY"

  APP_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$PLIST_PATH" 2>/dev/null || true)"
  [[ -n "$APP_BUNDLE_ID" ]] || die "unable to resolve CFBundleIdentifier"

  SIGNING_ENTITLEMENTS="$(mktemp)"
  if [[ -n "$PROVISIONING_PROFILE" ]]; then
    PROFILE_PLIST="$(mktemp)"
    security cms -D -i "$PROVISIONING_PROFILE" > "$PROFILE_PLIST" || die "unable to decode provisioning profile"
    PROFILE_PREFIX="$(/usr/libexec/PlistBuddy -c 'Print :ApplicationIdentifierPrefix:0' "$PROFILE_PLIST")" \
      || die "provisioning profile has no ApplicationIdentifierPrefix"
    PROFILE_TEAM_IDENTIFIER="$(/usr/libexec/PlistBuddy -c 'Print :TeamIdentifier:0' "$PROFILE_PLIST")" \
      || die "provisioning profile has no TeamIdentifier"
    PROFILE_APP_IDENTIFIER="$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:com.apple.application-identifier' "$PROFILE_PLIST")" \
      || die "provisioning profile has no application identifier"
    PROFILE_KEYCHAIN_GROUPS="$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:keychain-access-groups' "$PROFILE_PLIST")" \
      || die "provisioning profile has no Keychain access groups"
    PROFILE_EXPIRATION="$(plutil -extract ExpirationDate raw -o - "$PROFILE_PLIST")" \
      || die "provisioning profile has no expiration date"
    PROFILE_ALL_DEVICES="$(/usr/libexec/PlistBuddy -c 'Print :ProvisionsAllDevices' "$PROFILE_PLIST" 2>/dev/null || true)"
    rm -f "$PROFILE_PLIST"

    APP_IDENTIFIER="$PROFILE_PREFIX.$APP_BUNDLE_ID"
    [[ "$PROFILE_ALL_DEVICES" == "true" ]] || die "provisioning profile is not a Developer ID profile"
    [[ "$PROFILE_APP_IDENTIFIER" == "$APP_IDENTIFIER" ]] \
      || die "provisioning profile is for $PROFILE_APP_IDENTIFIER, expected $APP_IDENTIFIER"
    grep -qF -e "$APP_IDENTIFIER" -e "$PROFILE_PREFIX.*" <<< "$PROFILE_KEYCHAIN_GROUPS" \
      || die "provisioning profile does not allow the Keychain group $APP_IDENTIFIER"
    [[ "$(date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$PROFILE_EXPIRATION" '+%s')" -gt "$(date '+%s')" ]] \
      || die "provisioning profile expired on $PROFILE_EXPIRATION"

    cp "$PROVISIONING_PROFILE" "$APP_PATH/Contents/embedded.provisionprofile"
    sed \
      -e "s#\\\$(PRODUCT_BUNDLE_IDENTIFIER)#${APP_BUNDLE_ID}#g" \
      -e "s#\\\$(AppIdentifierPrefix)#${PROFILE_PREFIX}.#g" \
      -e "s#\\\$(DEVELOPMENT_TEAM)#${PROFILE_TEAM_IDENTIFIER}#g" \
      "$ROOT/Voxt/Voxt.entitlements" > "$SIGNING_ENTITLEMENTS"
  else
    sed "s#\\\$(PRODUCT_BUNDLE_IDENTIFIER)#${APP_BUNDLE_ID}#g" \
      "$ROOT/Voxt/Voxt.entitlements" > "$SIGNING_ENTITLEMENTS"
    # macOS refuses to launch an app that claims these entitlements without a
    # provisioning profile, so this signature drops them and the Keychain with them.
    /usr/libexec/PlistBuddy -c 'Delete :com.apple.application-identifier' "$SIGNING_ENTITLEMENTS"
    /usr/libexec/PlistBuddy -c 'Delete :com.apple.developer.team-identifier' "$SIGNING_ENTITLEMENTS"
    /usr/libexec/PlistBuddy -c 'Delete :keychain-access-groups' "$SIGNING_ENTITLEMENTS"
  fi
  if grep -qF '$(' "$SIGNING_ENTITLEMENTS"; then
    rm -f "$SIGNING_ENTITLEMENTS"
    die "failed to expand signing variables in entitlements"
  fi

  sign_if_exists() {
    local target="$1"
    if [[ -e "$target" ]]; then
      echo "Signing: $target"
      codesign --force --options runtime --timestamp --sign "$APP_SIGN_IDENTITY" "$target"
    fi
  }

  SPARKLE_FRAMEWORK="$APP_PATH/Contents/Frameworks/Sparkle.framework"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/Installer.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/InstallerLauncher.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/Downloader.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/Autoupdate"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/Updater.app"
  sign_if_exists "$SPARKLE_FRAMEWORK"

  while IFS= read -r dylib; do
    sign_if_exists "$dylib"
  done < <(find "$APP_PATH/Contents/Frameworks" -maxdepth 1 -type f -name '*.dylib' 2>/dev/null | sort)

  if [[ -n "$NATIVE_RUNTIME_DIR" ]]; then
    while IFS= read -r native_runtime_file; do
      if [[ "$(file -b "$native_runtime_file")" == *"Mach-O"* ]]; then
        sign_if_exists "$native_runtime_file"
      else
        echo "Skipping non-code runtime resource: $native_runtime_file"
      fi
    done < <(find "$APP_PATH/Contents/Resources/OmniRuntime/bin" -type f ! -name '*.metallib' -print | sort)
  fi

  while IFS= read -r framework_code; do
    if [[ "$(file -b "$framework_code")" == *"Mach-O"* ]]; then
      sign_if_exists "$framework_code"
    else
      echo "Skipping non-code framework resource: $framework_code"
    fi
  done < <(find "$APP_PATH/Contents/Frameworks" -type f -print | sort)

  while IFS= read -r framework_bundle; do
    sign_if_exists "$framework_bundle"
  done < <(find "$APP_PATH/Contents/Frameworks" -maxdepth 1 -type d -name '*.framework' -print | sort)

  codesign \
    --force \
    --options runtime \
    --timestamp \
    --entitlements "$SIGNING_ENTITLEMENTS" \
    --sign "$APP_SIGN_IDENTITY" \
    "$APP_PATH"

  rm -f "$SIGNING_ENTITLEMENTS"

  if [[ -n "$PROVISIONING_PROFILE" ]]; then
    SIGNED_ENTITLEMENTS="$(mktemp)"
    codesign --display --entitlements - --xml "$APP_PATH" > "$SIGNED_ENTITLEMENTS" 2>/dev/null
    SIGNED_KEYCHAIN_GROUP="$(/usr/libexec/PlistBuddy -c 'Print :keychain-access-groups:0' "$SIGNED_ENTITLEMENTS" 2>/dev/null || true)"
    rm -f "$SIGNED_ENTITLEMENTS"
    SIGNED_TEAM_IDENTIFIER="$(codesign --display --verbose=4 "$APP_PATH" 2>&1 | awk -F= '/^TeamIdentifier=/ {print $2; exit}')"
    [[ "$SIGNED_KEYCHAIN_GROUP" == "$APP_IDENTIFIER" ]] \
      || die "signed app has Keychain group '$SIGNED_KEYCHAIN_GROUP', expected $APP_IDENTIFIER"
    [[ "$SIGNED_TEAM_IDENTIFIER" == "$PROFILE_TEAM_IDENTIFIER" ]] \
      || die "signing certificate is from team '$SIGNED_TEAM_IDENTIFIER', but the provisioning profile is for $PROFILE_TEAM_IDENTIFIER"
  else
    echo "warning: no provisioning profile; the signed app cannot use the Keychain" >&2
  fi
elif [[ "$SIGN_APP" == "true" ]]; then
  echo "warning: no Developer ID Application identity found; app will remain unsigned" >&2
fi

if codesign --verify --strict --verbose=2 "$APP_PATH"; then
  echo "codesign verification passed"
else
  echo "warning: codesign verification failed; artifact was still packaged for local inspection" >&2
fi

if [[ "$NOTARIZE" == "true" ]]; then
  # Stapling the app itself lets it pass Gatekeeper offline from any of the artifacts.
  ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$OUTPUT_DIR/notarization.zip"
  notarize "$OUTPUT_DIR/notarization.zip"
  rm -f "$OUTPUT_DIR/notarization.zip"
  xcrun stapler staple "$APP_PATH"
fi

log "Creating artifacts in $OUTPUT_DIR"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"

rm -rf "$DMG_STAGING_DIR"
mkdir -p "$DMG_STAGING_DIR"
ditto "$APP_PATH" "$DMG_STAGING_DIR/$APP_NAME.app"
ln -s /Applications "$DMG_STAGING_DIR/Applications"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$DMG_STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"
rm -rf "$DMG_STAGING_DIR"
if [[ "$SIGN_APP" == "true" && -n "$APP_SIGN_IDENTITY" ]]; then
  codesign --force --timestamp --sign "$APP_SIGN_IDENTITY" "$DMG_PATH"
fi

PKG_ARGS=(--component "$APP_PATH" /Applications)
if [[ -n "$INSTALLER_SIGN_IDENTITY" ]]; then
  PKG_ARGS+=(--sign "$INSTALLER_SIGN_IDENTITY")
fi
PKG_ARGS+=("$PKG_PATH")
productbuild "${PKG_ARGS[@]}"

if [[ "$NOTARIZE" == "true" ]]; then
  notarize "$DMG_PATH"
  notarize "$PKG_PATH"
  xcrun stapler staple "$DMG_PATH"
  xcrun stapler staple "$PKG_PATH"
  spctl --assess --type execute "$APP_PATH" || die "Gatekeeper rejects $APP_PATH"
  spctl --assess --type open --context context:primary-signature "$DMG_PATH" || die "Gatekeeper rejects $DMG_PATH"
  spctl --assess --type install "$PKG_PATH" || die "Gatekeeper rejects $PKG_PATH"
  echo "Gatekeeper accepts the app, the DMG and the PKG"
fi

for artifact in "$ZIP_PATH" "$DMG_PATH" "$PKG_PATH"; do
  (cd "$OUTPUT_DIR" && shasum -a 256 "$(basename "$artifact")" > "$(basename "$artifact").sha256")
done

if [[ "$INSTALL_APP" == "true" ]]; then
  log "Installing $APP_NAME.app into $INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"
  osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
  rm -rf "$INSTALL_DIR/$APP_NAME.app"
  ditto "$APP_PATH" "$INSTALL_DIR/$APP_NAME.app"
  xattr -dr com.apple.quarantine "$INSTALL_DIR/$APP_NAME.app" >/dev/null 2>&1 || true

  if [[ "$OPEN_AFTER_INSTALL" == "true" ]]; then
    open "$INSTALL_DIR/$APP_NAME.app"
  fi
fi

cat <<EOF

Done.
App: $APP_PATH
Zip: $ZIP_PATH
Dmg: $DMG_PATH
Pkg: $PKG_PATH
EOF
