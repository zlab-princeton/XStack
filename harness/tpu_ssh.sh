#!/usr/bin/env bash
# Run a command on the eval TPU VM (worker 0), non-interactively.
# usage: tpu_ssh.sh 'command ...'
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tpu_env.sh"
IP="$(tpu_ip)"
[ -n "$IP" ] || { echo "tpu_ssh: no address for the TPU VM (XSTACK_TPU_NAME and XSTACK_TPU_ZONE, or XSTACK_TPU_IPS, in config.env)" >&2; exit 2; }
exec ssh "${TPU_SSH_OPTS[@]}" "$XSTACK_TPU_USER@$IP" "$@"
