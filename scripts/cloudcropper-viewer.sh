#!/usr/bin/env bash
# Launch the CloudCropper viewer. With no argument it opens an empty window;
# load a cloud by dragging a .ply/.pcd/.npz file onto it, the "Open…" button,
# or the path box. Builds the gui preset on first run if needed.
set -e
cd "$(dirname "$0")/.."
export VCPKG_ROOT="$PWD/third_party/vcpkg"

BIN="build/gui/src/app/cloudcropper"
if [ ! -x "$BIN" ]; then
    echo "building the gui preset (first run)…" >&2
    cmake --preset gui
    cmake --build --preset gui
fi

exec "$BIN" view "$@"
