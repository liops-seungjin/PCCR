#!/usr/bin/env bash
# Rebuild and restart the CloudCropper viewer whenever source files change.
# Usage:
#   ./scripts/cloudcropper-viewer-watch.sh [cloud.ply|cloud.npz|...]
set -euo pipefail

cd "$(dirname "$0")/.."
export VCPKG_ROOT="$PWD/third_party/vcpkg"

BIN="build/gui/src/app/cloudcropper"
WATCH_PATHS=(
    CMakeLists.txt
    CMakePresets.json
    vcpkg.json
    include
    src
    backend
    config
    scripts/cloudcropper-viewer.sh
    scripts/cloudcropper-viewer-watch.sh
)

viewer_pid=""

existing_watch_paths() {
    local p
    for p in "${WATCH_PATHS[@]}"; do
        [[ -e "$p" ]] && printf '%s\n' "$p"
    done
}

configure_if_needed() {
    if [[ ! -f build/gui/build.ninja ]]; then
        echo "configuring gui preset..." >&2
        cmake --preset gui
    fi
}

build_viewer() {
    configure_if_needed
    cmake --build --preset gui
}

start_viewer() {
    if [[ ! -x "$BIN" ]]; then
        echo "viewer binary missing after build: $BIN" >&2
        return 1
    fi
    echo "starting viewer: $BIN view $*" >&2
    "$BIN" view "$@" &
    viewer_pid=$!
}

stop_viewer() {
    [[ -n "${viewer_pid:-}" ]] || return 0
    if kill -0 "$viewer_pid" 2>/dev/null; then
        echo "stopping viewer pid $viewer_pid" >&2
        kill "$viewer_pid" 2>/dev/null || true
        for _ in {1..40}; do
            if ! kill -0 "$viewer_pid" 2>/dev/null; then
                wait "$viewer_pid" 2>/dev/null || true
                viewer_pid=""
                return 0
            fi
            sleep 0.1
        done
        kill -9 "$viewer_pid" 2>/dev/null || true
        wait "$viewer_pid" 2>/dev/null || true
    fi
    viewer_pid=""
}

fingerprint() {
    mapfile -t paths < <(existing_watch_paths)
    find "${paths[@]}" \
        \( -path './build' -o -path './.git' -o -path './third_party/vcpkg/buildtrees' \
           -o -path './third_party/vcpkg/downloads' \) -prune -o \
        -type f \
        \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' -o \
           -name '*.cmake' -o -name 'CMakeLists.txt' -o -name 'CMakePresets.json' -o \
           -name 'vcpkg.json' -o -name '*.yaml' -o -name '*.py' -o -name '*.sh' \) \
        -printf '%T@ %s %p\n' 2>/dev/null | sort
}

wait_for_change_inotify() {
    mapfile -t paths < <(existing_watch_paths)
    inotifywait -q -r \
        -e close_write,modify,create,delete,move \
        --exclude '(^|/)(build|\.git|third_party/vcpkg/buildtrees|third_party/vcpkg/downloads)(/|$)' \
        "${paths[@]}" >/dev/null
}

wait_for_change_polling() {
    local before after
    before="$(fingerprint)"
    while true; do
        sleep 1
        after="$(fingerprint)"
        [[ "$before" != "$after" ]] && return 0
    done
}

wait_for_change() {
    echo "watching for source changes..." >&2
    if command -v inotifywait >/dev/null 2>&1; then
        wait_for_change_inotify
    else
        echo "inotifywait not found; polling every 1s (install inotify-tools for faster watching)" >&2
        wait_for_change_polling
    fi
}

cleanup() {
    stop_viewer
}
trap cleanup EXIT INT TERM

while true; do
    if build_viewer; then
        start_viewer "$@" || true
    else
        echo "build failed; fix the error and save a watched file to retry" >&2
    fi
    wait_for_change
    stop_viewer
done
