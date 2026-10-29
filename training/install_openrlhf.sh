#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TARGET=${1:-"$ROOT/vendor/OpenRLHF"}
REVISION=cf69bd61a5fbcfa4f07bad2af76797643ac557e5
if [[ -e "$TARGET" ]]; then
    echo "Destination already exists: $TARGET. Choose a new directory." >&2
    exit 1
fi
git clone https://github.com/OpenRLHF/OpenRLHF.git "$TARGET"
git -C "$TARGET" checkout --detach "$REVISION"
cp -R "$ROOT/training/openrlhf_overlay/openrlhf/." "$TARGET/openrlhf/"
echo "Prepared OpenRLHF $REVISION with TIPCoder modifications in $TARGET"
echo "Install in a dedicated GPU environment: python -m pip install -e '$TARGET'"
