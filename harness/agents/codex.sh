#!/usr/bin/env bash
# The Codex launcher: one turn of `codex exec` inside a read-masking sandbox (bwrap). The agent sees
# its workspace, its own agent home, its machine key and the host's tool installs, and nothing of
# this repository (the hidden criteria), the work directory (the ledger, other cells) or the
# operator's home. Codex's own command sandbox is turned off inside: the agent's commands must reach
# its TPU VM over ssh, and the read mask is the isolation.
#
# Drop-in for the codex binary: every argument is forwarded, `--cd DIR` names the workspace.
# XSTACK_NO_SANDBOX=1 runs without the mask (debugging only: the hidden criteria are then readable).
set -uo pipefail

WORKDIR=""; FWD=(); PREV=""
for arg in "$@"; do
  if [ "$PREV" = "--sandbox" ]; then FWD+=("danger-full-access"); PREV=""; continue; fi
  FWD+=("$arg")
  [ "$PREV" = "--cd" ] && WORKDIR="$arg"
  [ "$arg" = "exec" ] && FWD+=("--skip-git-repo-check")   # the masked view hides the repository's .git
  PREV="$arg"
done
[ -n "$WORKDIR" ] || { echo "codex launcher: missing --cd" >&2; exit 2; }
[ -d "$WORKDIR" ] || mkdir -p "$WORKDIR"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${XSTACK_WORK:-$REPO/work}"
# Memories off: Codex would inject memories of earlier sessions on this host, keyed by cwd.
# Reasoning summaries on: the transcript then carries the model's thinking summaries.
CONF=(-c 'features.memories=false' -c 'model_reasoning_summary="detailed"')

if [ "${XSTACK_NO_SANDBOX:-0}" = 1 ]; then
  exec codex "${CONF[@]}" "${FWD[@]}" < /dev/null
fi
command -v bwrap >/dev/null 2>&1 || { echo "codex launcher: bwrap not found (install bubblewrap; XSTACK_NO_SANDBOX=1 runs unmasked, for debugging only)" >&2; exit 3; }

AGENT_TMP="${XSTACK_TMP:-$WORK/tmp}/agent-tmp/$(basename "$WORKDIR")-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$AGENT_TMP"
AGENT_SSH="$(dirname "$WORKDIR")/agent_ssh"   # the cell's own key: it opens the cell's machine only
BINDS=(--tmpfs "$HOME")                       # the operator's home is masked, tool installs come back read-only
for d in .nvm .local; do [ -d "$HOME/$d" ] && BINDS+=(--ro-bind "$HOME/$d" "$HOME/$d"); done
BINDS+=(--tmpfs "$REPO" --tmpfs "$WORK" --tmpfs "$HOME/.config/gcloud" --tmpfs "$HOME/.ssh")
[ -d "$AGENT_SSH" ] && BINDS+=(--bind "$AGENT_SSH" "$HOME/.ssh")
[ -n "${CODEX_HOME:-}" ] && BINDS+=(--dev-bind "$CODEX_HOME" "$CODEX_HOME")
BINDS+=(--dev-bind "$WORKDIR" "$WORKDIR")
# Own PID namespace: whatever the agent detaches dies with the turn.
exec bwrap --unshare-pid --die-with-parent --dev-bind / / --proc /proc \
  --bind "$AGENT_TMP" /tmp --setenv TMPDIR /tmp \
  --setenv UV_CACHE_DIR /tmp/uv-cache --setenv XDG_CACHE_HOME /tmp/xdg-cache --setenv PIP_CACHE_DIR /tmp/pip-cache \
  "${BINDS[@]}" \
  -- codex "${CONF[@]}" "${FWD[@]}" < /dev/null
