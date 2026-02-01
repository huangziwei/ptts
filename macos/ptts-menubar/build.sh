#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
APP_NAME="pTTS Menubar"
APP_DIR="${BUILD_DIR}/${APP_NAME}.app"
BIN_NAME="ptts-menubar"

mkdir -p "${BUILD_DIR}"

swiftc -O -framework Cocoa \
  "${ROOT_DIR}/src/main.swift" \
  -o "${BUILD_DIR}/${BIN_NAME}"

mkdir -p "${APP_DIR}/Contents/MacOS" "${APP_DIR}/Contents/Resources"
cp "${ROOT_DIR}/Info.plist" "${APP_DIR}/Contents/Info.plist"
cp "${BUILD_DIR}/${BIN_NAME}" "${APP_DIR}/Contents/MacOS/${BIN_NAME}"

codesign --force --sign - "${APP_DIR}" >/dev/null 2>&1 || true

echo "Built: ${APP_DIR}"
