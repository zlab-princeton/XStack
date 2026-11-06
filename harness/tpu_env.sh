# Shared configuration of the XStack harness (source me, don't run me).
#
# Reads config.env at the repository root (copy config.env.example), derives every work location from
# XSTACK_WORK, and resolves the TPU VM from XSTACK_TPU_NAME and XSTACK_TPU_ZONE (gcloud) or
# XSTACK_TPU_IPS. One standard single-host TPU VM per concurrent cell; the operator reaches it with
# gcloud by name, the agent with its own per-cell ssh key.
_XSTACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$_XSTACK_ROOT/config.env" ] && . "$_XSTACK_ROOT/config.env"
export XSTACK_WORK="${XSTACK_WORK:-$_XSTACK_ROOT/work}"
export XSTACK_TMP="$XSTACK_WORK/tmp"
mkdir -p "$XSTACK_WORK" "$XSTACK_TMP" 2>/dev/null
export TMPDIR="$XSTACK_TMP" TMP="$XSTACK_TMP" TEMP="$XSTACK_TMP"
export XSTACK_ROWS="$XSTACK_WORK/ledger.jsonl"
XSTACK_TPU_ZONE="${XSTACK_TPU_ZONE:-}"
export XSTACK_TPU_USER="${XSTACK_TPU_USER:-$USER}"
XSTACK_TPU_KEY="${XSTACK_TPU_KEY:-$HOME/.ssh/id_rsa}"
XSTACK_TPU_KEY="${XSTACK_TPU_KEY/#\~/$HOME}"

tpu_name() { printf '%s\n' "${XSTACK_TPU_NAME:-}"; }
tpu_zone() { printf '%s\n' "${XSTACK_TPU_ZONE:-}"; }
tpu_ips() {  # the worker external IPs, one per line (a single-host VM has one)
  if [ -n "${XSTACK_TPU_IPS:-}" ]; then printf '%s\n' $XSTACK_TPU_IPS; return 0; fi
  [ -n "${XSTACK_TPU_NAME:-}" ] && [ -n "${XSTACK_TPU_ZONE:-}" ] || return 0
  gcloud compute tpus tpu-vm describe "$XSTACK_TPU_NAME" --zone "$XSTACK_TPU_ZONE" \
    --format='value(networkEndpoints[].accessConfig.externalIp)' 2>/dev/null | tail -1 | tr ';' '\n' | tr -d ' ' | sed '/^$/d'
}
tpu_ip() { tpu_ips | head -1; }
tpu_accel() { printf '%s\n' "${XSTACK_TPU_ACCEL:-$(tpu_name | grep -Eo 'v[0-9]+[a-z]*-[0-9]+' | head -1)}"; }
tpu_chips() {  # chips on the VM: v5e/v6e names count chips, v4/v5p names count TensorCores (2 per chip)
  local a n; a="$(tpu_accel)"; n="${a##*-}"
  case "$a" in v4-*|v5p-*) echo $(( n / 2 ));; "") echo "";; *) echo "$n";; esac
}
# The operator's transport to the machine: gcloud by name, worker 0.
tpu_gssh() {  # tpu_gssh TIMEOUT CMD
  timeout "$1" gcloud alpha compute tpus tpu-vm ssh "$XSTACK_TPU_USER@$(tpu_name)" \
    --zone "$(tpu_zone)" --worker=0 \
    --ssh-flag='-o ConnectTimeout=45' --ssh-flag='-o StrictHostKeyChecking=no' \
    --ssh-flag='-o UserKnownHostsFile=/dev/null' \
    --command "$2" --quiet \
    --ssh-flag=-i --ssh-flag="$XSTACK_TPU_KEY" 2>/dev/null
}
tpu_gscp_to() {  # tpu_gscp_to LOCAL REMOTE [TIMEOUT]
  timeout "${3:-420}" gcloud alpha compute tpus tpu-vm scp --recurse "$1" \
    "$XSTACK_TPU_USER@$(tpu_name):$2" --zone "$(tpu_zone)" --worker=0 \
    --quiet --scp-flag=-i --scp-flag="$XSTACK_TPU_KEY" >/dev/null 2>&1
}
tpu_gscp_from() {  # tpu_gscp_from REMOTE LOCAL [TIMEOUT]
  timeout "${3:-420}" gcloud alpha compute tpus tpu-vm scp --recurse \
    "$XSTACK_TPU_USER@$(tpu_name):$1" "$2" --zone "$(tpu_zone)" --worker=0 \
    --quiet --scp-flag=-i --scp-flag="$XSTACK_TPU_KEY" >/dev/null 2>&1
}
# ssh/scp option sets for tpu_ssh.sh and tpu_scp.sh (plain ssh to the address).
TPU_SSH_OPTS=(-i "$XSTACK_TPU_KEY" -o StrictHostKeyChecking=no
              -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30)
