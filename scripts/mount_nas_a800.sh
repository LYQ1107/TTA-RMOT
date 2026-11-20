#!/usr/bin/env bash
set -euo pipefail
nas_a800_mount=/data3/liuyeqiang/nas/A800
if mountpoint -q "$nas_a800_mount"; then
    exit 0
fi
mkdir -p "$nas_a800_mount"
if [ -n "$(ls -A "$nas_a800_mount")" ]; then
    echo "NAS mount directory contains local files: $nas_a800_mount" >&2
    exit 1
fi
exec /data3/liuyeqiang/.local/share/nas-mount/runtime/usr/bin/sshfs \
    Valadmin@127.0.0.1:/home/A800 "$nas_a800_mount" -p 24557 \
    -o ro,reconnect,default_permissions,nosuid,nodev \
    -o uid=1002,gid=1002,umask=0077,attr_timeout=20,entry_timeout=20 \
    -o IdentityFile=/data3/liuyeqiang/.ssh/id_ed25519_nas_readwrite \
    -o IdentitiesOnly=yes,BatchMode=yes,StrictHostKeyChecking=yes \
    -o HostKeyAlias=nas-readwrite \
    -o UserKnownHostsFile=/data3/liuyeqiang/.ssh/known_hosts_nas_readwrite \
    -o ServerAliveInterval=15,ServerAliveCountMax=2,ConnectTimeout=10
