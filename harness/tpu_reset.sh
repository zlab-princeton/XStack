#!/usr/bin/env bash
# Wipe the TPU VM's user state between cells, and between a turn and its grade: leftover processes, the
# home directory (system dotfiles and ~/.ssh kept), /tmp and /dev/shm. The TPU itself is untouched.
# "Fresh machine" in the prompt means this. Transport: gcloud by name (harness/tpu_env.sh).
set -uo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tpu_env.sh"
[ -n "$(tpu_name)" ] || { echo "tpu_reset: XSTACK_TPU_NAME is not set (config.env)" >&2; exit 2; }
SWEEP='
  pkill -9 -f "pytho[n]" 2>/dev/null; sleep 1
  cd ~ || exit 1
  for f in * .[!.]*; do
    case "$f" in
      .bashrc|.bash_logout|.profile|.zshrc|.ssh|.config) ;;
      *) rm -rf -- "$f";;
    esac
  done
  sed -i "/uvbin\/env/d" ~/.bashrc ~/.profile ~/.zshrc 2>/dev/null
  # logind removes the user'"'"'s /dev/shm files when the last ssh session ends; a graded run keeps
  # caches there, so linger is enabled once per machine.
  if [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" != "yes" ]; then sudo -n loginctl enable-linger "$(id -un)" 2>/dev/null || true; fi
  find /dev/shm /tmp /var/tmp -maxdepth 1 -user "$(id -un)" ! -path /dev/shm ! -path /tmp ! -path /var/tmp -exec rm -rf {} + 2>/dev/null
  echo VM_RESET "$(hostname)"
'
tpu_gssh 150 "$SWEEP" || exit 1
