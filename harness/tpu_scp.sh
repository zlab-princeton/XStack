#!/usr/bin/env bash
# Copy files to/from the eval TPU VM. Remote paths are written with a leading
# colon:  tpu_scp.sh local_dir :/remote/path   or   tpu_scp.sh :/remote/f .
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tpu_env.sh"
IP="$(tpu_ip)"
[ -n "$IP" ] || { echo "tpu_scp: no address for the TPU VM (XSTACK_TPU_NAME and XSTACK_TPU_ZONE, or XSTACK_TPU_IPS, in config.env)" >&2; exit 2; }
ARGS=()
for a in "$@"; do
  case "$a" in
    :*) ARGS+=("$XSTACK_TPU_USER@$IP$a");;
    *) ARGS+=("$a");;
  esac
done
exec scp "${TPU_SSH_OPTS[@]}" -r "${ARGS[@]}"
