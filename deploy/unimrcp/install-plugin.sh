#!/usr/bin/env bash
#
# Install the sandroid-recog UniMRCP plugin on a host where UniMRCP 1.8.0
# has already been built and installed via deploy/ubuntu/unimrcp-setup.md.
#
# Steps:
#   1. rsync native/unimrcp-plugin/ into $UNIMRCP_SRC/plugins/sandroid-recog/
#   2. apply patch 0002 to register the plugin in configure.ac + plugins/Makefile.am
#   3. bootstrap + configure + make + make install
#   4. apply patch 0003 to point speechrecog at Sandroid-Recog-1 in
#      /opt/unimrcp/conf/unimrcpserver.xml
#
# Idempotent: safe to rerun after local edits. Patches are applied with
# `patch --forward` so repeated runs do not double-apply.
#
# Usage:
#   UNIMRCP_SRC=/opt/unimrcp-src \
#   UNIMRCP_PREFIX=/opt/unimrcp \
#   SANDROID_REPO=/path/to/sandroid \
#   ./install-plugin.sh

set -euo pipefail

UNIMRCP_SRC="${UNIMRCP_SRC:-/opt/unimrcp-src}"
UNIMRCP_PREFIX="${UNIMRCP_PREFIX:-/opt/unimrcp}"
SANDROID_REPO="${SANDROID_REPO:-$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)}"

PATCH_DIR="$SANDROID_REPO/deploy/unimrcp/patches"
PLUGIN_SRC="$SANDROID_REPO/native/unimrcp-plugin"
PLUGIN_DST="$UNIMRCP_SRC/plugins/sandroid-recog"
SERVER_XML="$UNIMRCP_PREFIX/conf/unimrcpserver.xml"

die() { echo "error: $*" >&2; exit 1; }

[[ -d "$UNIMRCP_SRC" ]]   || die "UNIMRCP_SRC=$UNIMRCP_SRC does not exist"
[[ -d "$PLUGIN_SRC" ]]    || die "plugin source not found at $PLUGIN_SRC"
[[ -f "$SERVER_XML" ]]    || die "unimrcpserver.xml missing; is UniMRCP installed?"

echo "==> Sync plugin sources -> $PLUGIN_DST"
mkdir -p "$PLUGIN_DST"
rsync -a --delete "$PLUGIN_SRC/" "$PLUGIN_DST/"

echo "==> Apply build-system patch (configure.ac + plugins/Makefile.am)"
cd "$UNIMRCP_SRC"
patch -p1 --forward --silent < "$PATCH_DIR/0002-add-sandroid-recog-plugin.patch" || true

echo "==> Rebuild + install UniMRCP"
./bootstrap
./configure --prefix="$UNIMRCP_PREFIX" --with-sofia-sip=/usr
make -j"$(nproc)"
make install

echo "==> Apply runtime config patch (unimrcpserver.xml)"
cd "$(dirname "$SERVER_XML")"
patch -p0 --forward --silent < "$PATCH_DIR/0003-switch-speechrecog-to-sandroidrecog.patch" || true

# Optional: install the Python bridge systemd unit. Set INSTALL_BRIDGE=1 to enable.
if [[ "${INSTALL_BRIDGE:-0}" == "1" ]]; then
    SANDROID_PREFIX="${SANDROID_PREFIX:-/opt/sandroid}"
    echo "==> Stage Python bridge sources -> $SANDROID_PREFIX/src"
    mkdir -p "$SANDROID_PREFIX/src" "$SANDROID_PREFIX/logs"
    rsync -a --delete \
        --include='sandroid/' \
        --include='sandroid/**' \
        --exclude='*' \
        "$SANDROID_REPO/src/" "$SANDROID_PREFIX/src/"

    echo "==> Install systemd unit -> /etc/systemd/system/sandroid-bridge.service"
    install -m 0644 "$SANDROID_REPO/deploy/unimrcp/bridge.service" \
        /etc/systemd/system/sandroid-bridge.service
    systemctl daemon-reload
    systemctl enable --now sandroid-bridge.service
    systemctl --no-pager status sandroid-bridge.service | head -10 || true
fi

echo
echo "Done. To start the server:"
echo "  LD_LIBRARY_PATH=$UNIMRCP_PREFIX/lib $UNIMRCP_PREFIX/bin/unimrcpserver -r $UNIMRCP_PREFIX -d"
echo
echo "Expected log lines:"
echo "  Load Plugin [Sandroid-Recog-1] [$UNIMRCP_PREFIX/plugin/sandroidrecog.so]"
echo "  Associate Resource [speechrecog] to Engine [Sandroid-Recog-1]"
echo
echo "If INSTALL_BRIDGE=1 was set, verify the Python bridge:"
echo "  systemctl status sandroid-bridge"
echo "  ls -l /var/run/sandroid/bridge.sock"
