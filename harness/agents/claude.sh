#!/usr/bin/env bash
# The Claude Code launcher: one turn of `claude -p`, for Anthropic models and for any provider with an
# Anthropic-compatible endpoint (GLM, DeepSeek, Muse, Kimi ...), inside the same read-masking sandbox
# as the Codex launcher and with a private config directory per cell, so the operator's own Claude
# Code setup is never read or written.
#
# CLI surface: the Codex one, so the driver calls both launchers the same way:
#   claude.sh [-c model_reasoning_effort="E"] [--ask-for-approval never] exec --cd DIR [--sandbox X]
#             --output-last-message FILE --model LABEL [resume SID] PROMPT
#
# Provider file: $XSTACK_WORK/secrets/<provider>.env, picked by the model label (claude* -> anthropic,
# glm* -> zai, deepseek* -> deepseek, muse* -> meta, kimi* -> kimi, else <label>.env). It exports
#   XSTACK_AGENT_TOKEN        the API key or token. "subscription" uses the operator's own `claude login`;
#                             an Anthropic OAuth token (sk-ant-oat01-...) goes in as CLAUDE_CODE_OAUTH_TOKEN
#   XSTACK_AGENT_BASE_URL     the provider's Anthropic-compatible endpoint (unset for Anthropic itself)
#   XSTACK_AGENT_MODEL        the provider's model id for the label (default: the label)
#   XSTACK_AGENT_DISALLOWED   optional value for claude's --disallowedTools
# XSTACK_NO_SANDBOX=1 runs without the mask (debugging only), as for the Codex launcher.
# The transcript is the stream-json trace: claude-trace.jsonl in the workspace for the driver and a
# cumulative per-cell file in the agent home, hard-linked as sessions/rollout-<session>.jsonl so the
# driver finds it the way it finds a Codex rollout.
# Exit: claude's own; 97 the stream went silent after real activity (the turn fails, ungraded);
# 124 it went silent before any model output (infrastructure: the driver retries the turn).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${XSTACK_WORK:-$REPO/work}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- the Codex-style arguments ---------------------------------------------------------------
EFFORT=""; WORKDIR=""; OUTFILE=""; MODEL=""; PROMPT=""; RESUME_SID=""
while [ $# -gt 0 ]; do
  case "$1" in
    -c) case "$2" in model_reasoning_effort=*) EFFORT="${2#model_reasoning_effort=}"; EFFORT="${EFFORT//\"/}";; esac; shift 2;;
    --ask-for-approval|--sandbox|--cd|--output-last-message|--model|resume)
      case "$1" in --cd) WORKDIR="$2";; --output-last-message) OUTFILE="$2";; --model) MODEL="$2";; resume) RESUME_SID="$2";; esac
      shift 2;;
    exec|--ephemeral) shift;;
    *) PROMPT="$1"; shift;;
  esac
done
[ -n "$WORKDIR" ] || { echo "claude launcher: missing --cd" >&2; exit 2; }
[ -n "$PROMPT" ] || { echo "claude launcher: missing prompt" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "claude launcher: missing --model" >&2; exit 2; }

# ---- the provider ------------------------------------------------------------------------------
case "$MODEL" in
  claude*|Claude*|anthropic*) PROVIDER=anthropic;;
  glm*|GLM*|zai*) PROVIDER=zai;;
  deepseek*|DeepSeek*) PROVIDER=deepseek;;
  muse*|Muse*|meta*) PROVIDER=meta;;
  kimi*|Kimi*|moonshot*) PROVIDER=kimi;;
  *) PROVIDER="$MODEL";;
esac
SECRETS="$WORK/secrets/$PROVIDER.env"
[ -f "$SECRETS" ] || { echo "claude launcher: missing $SECRETS (see README, Agents)" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SECRETS"
: "${XSTACK_AGENT_TOKEN:?$SECRETS must export XSTACK_AGENT_TOKEN}"
MODEL_ID="${XSTACK_AGENT_MODEL:-$MODEL}"

# ---- a private config directory per cell: the session store lives in the agent home ------------
if [ -n "${CODEX_HOME:-}" ]; then
  CFG="$CODEX_HOME/claude_cfg"
else
  mkdir -p "$WORK/claude_homes" && chmod 700 "$WORK/claude_homes"
  CFG="$(mktemp -d "$WORK/claude_homes/cfg.XXXXXXXX")"
fi
mkdir -p "$CFG" && chmod 700 "$CFG"
export CLAUDE_CONFIG_DIR="$CFG"
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY 2>/dev/null || true
if [[ "$XSTACK_AGENT_TOKEN" == sk-ant-oat01-* ]]; then
  export CLAUDE_CODE_OAUTH_TOKEN="$XSTACK_AGENT_TOKEN"   # a long-lived token from `claude setup-token`
elif [ "$XSTACK_AGENT_TOKEN" = "subscription" ]; then
  CRED="$HOME/.claude/.credentials.json"                 # the operator's own login, copied into the cell's config dir
  [ -f "$CRED" ] || { echo "claude launcher: XSTACK_AGENT_TOKEN=subscription but $CRED is missing (run claude login)" >&2; exit 2; }
  cp "$CRED" "$CFG/.credentials.json" && chmod 600 "$CFG/.credentials.json"
else
  if [ -n "${XSTACK_AGENT_BASE_URL:-}" ] && [[ "$XSTACK_AGENT_BASE_URL" != *api.anthropic.com* ]]; then
    export ANTHROPIC_BASE_URL="$XSTACK_AGENT_BASE_URL" ANTHROPIC_AUTH_TOKEN="$XSTACK_AGENT_TOKEN"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_ID" ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ID" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_ID"
  else
    export ANTHROPIC_API_KEY="$XSTACK_AGENT_TOKEN"
  fi
fi
export API_TIMEOUT_MS=3000000 DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

CMD=(claude -p "$PROMPT" --dangerously-skip-permissions --model "$MODEL_ID")
[ -n "${XSTACK_AGENT_DISALLOWED-}" ] && CMD+=(--disallowedTools "$XSTACK_AGENT_DISALLOWED")
[ -n "$EFFORT" ] && CMD+=(--effort "$EFFORT")
if [ -n "$RESUME_SID" ]; then   # claude may rotate the session id across resumes: the latest one wins
  REAL_SID="$(cat "${CODEX_HOME:-$CFG}/claude_last_session_id" 2>/dev/null || true)"
  CMD+=(--resume "${REAL_SID:-$RESUME_SID}")
fi
CMD+=(--output-format stream-json --verbose)

# ---- the sandbox: the same masks as the Codex launcher, plus the operator's ~/.claude -------------
if [ "${XSTACK_NO_SANDBOX:-0}" != 1 ]; then
  command -v bwrap >/dev/null 2>&1 || { echo "claude launcher: bwrap not found (install bubblewrap; XSTACK_NO_SANDBOX=1 runs unmasked, for debugging only)" >&2; exit 3; }
  AGENT_TMP="${XSTACK_TMP:-$WORK/tmp}/agent-tmp/$(basename "$WORKDIR")-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir -p "$AGENT_TMP"
  AGENT_SSH="$(dirname "$WORKDIR")/agent_ssh"
  BINDS=(--tmpfs "$HOME")
  for d in .nvm .local; do [ -d "$HOME/$d" ] && BINDS+=(--ro-bind "$HOME/$d" "$HOME/$d"); done
  BINDS+=(--tmpfs "$REPO" --tmpfs "$WORK" --tmpfs "$HOME/.claude" --tmpfs "$HOME/.config/gcloud" --tmpfs "$HOME/.ssh")
  [ -d "$AGENT_SSH" ] && BINDS+=(--bind "$AGENT_SSH" "$HOME/.ssh")
  [ -n "${CODEX_HOME:-}" ] && BINDS+=(--dev-bind "$CODEX_HOME" "$CODEX_HOME")
  case "$CFG" in "$WORK"/*) BINDS+=(--dev-bind "$CFG" "$CFG");; esac
  BINDS+=(--dev-bind "$WORKDIR" "$WORKDIR")
  CMD=(bwrap --unshare-pid --die-with-parent --dev-bind / / --proc /proc
       --bind "$AGENT_TMP" /tmp --setenv TMPDIR /tmp
       --setenv UV_CACHE_DIR /tmp/uv-cache --setenv XDG_CACHE_HOME /tmp/xdg-cache --setenv PIP_CACHE_DIR /tmp/pip-cache
       "${BINDS[@]}" -- "${CMD[@]}")
fi

# ---- the run: the trace, the session id, the idle watchdog, the provider usage cap ----------------
cd "$WORKDIR" || exit 2
if [ -z "${CODEX_HOME:-}" ]; then   # not from the driver: run once, no bookkeeping
  "${CMD[@]}" > claude-trace.jsonl 2> shim-claude-stderr.txt
  STATUS=$?
  TRACE_SRC=claude-trace.jsonl; TRACE_OFF=0
else
  CUM="$CODEX_HOME/claude_trace_cum.jsonl"   # every turn of the cell, in order; read by the watchers below
  touch "$CUM"
  TRACE_SRC="$CUM"; BASE_OFF=$(stat -c %s "$CUM"); TRACE_OFF=$BASE_OFF; RUN_OFF=$BASE_OFF
  IDLE_TMO=1800   # seconds of silence after real activity that end the turn as a hang
  HUNG_MARKER="$WORKDIR/.shim_idle_killed"
  rm -f "$HUNG_MARKER"
  : > shim-claude-stderr.txt
  run_claude_once() {
    RUN_OFF=$(stat -c %s "$CUM")
    (  # the session id, as soon as the stream's first event lands; the driver's rollout name is a hard link
      for _i in $(seq 1 120); do
        s="$(tail -c +$((RUN_OFF+1)) "$CUM" 2>/dev/null | head -1 | grep -oE '"session_id"[": ]+[0-9a-f-]{36}' | grep -oE '[0-9a-f-]{36}')"
        if [ -n "$s" ]; then
          printf '%s\n' "$s" > "$CODEX_HOME/claude_last_session_id"
          mkdir -p "$CODEX_HOME/sessions"
          [ -e "$CODEX_HOME/sessions/rollout-$s.jsonl" ] || ln "$CUM" "$CODEX_HOME/sessions/rollout-$s.jsonl" 2>/dev/null || true
          exit 0
        fi
        sleep 1
      done
    ) &
    SID_WATCHER=$!
    (  # the idle watchdog: a stream silent for IDLE_TMO is killed, unless the agent parked itself with a
       # scheduled wakeup (shim_park_pending.py reads the pending delay); then the kill waits for the
       # wakeup to be overdue
      while sleep 60; do
        age=$(( $(date +%s) - $(stat -c %Y "$CUM" 2>/dev/null || date +%s) ))
        if [ "$age" -ge "$IDLE_TMO" ]; then
          park="$(python3 "$HERE/../shim_park_pending.py" "$CUM" 2>/dev/null || echo 0)"; park="${park:-0}"
          if [ "$park" -gt 0 ] && [ "$age" -lt $((park + 600)) ]; then
            printf 'park_s=%s age=%s at=%s\n' "$park" "$age" "$(date -u +%FT%TZ)" >> "$WORKDIR/.shim_parked"
            continue
          fi
          touch "$HUNG_MARKER"
          for p in $(pgrep -u "$(id -un)" -f 'claude -[p]'); do
            [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$WORKDIR" ] && kill -9 "$p" 2>/dev/null
          done
          exit 0
        fi
      done
    ) &
    IDLE_WATCHER=$!
    "${CMD[@]}" 2>> shim-claude-stderr.txt | tee -a "$CUM" > claude-trace.jsonl
    STATUS=${PIPESTATUS[0]}
    kill "$IDLE_WATCHER" 2>/dev/null
    wait "$SID_WATCHER" 2>/dev/null || true
  }
  run_claude_once
  # A provider usage cap ends the stream as an error result naming a reset time. That is the plan's
  # meter, not the agent: wait for the reset, then resume the session with this turn's prompt, at
  # most CAP_MAX times. Each wait is recorded in .shim_cap_waits.
  CAP_WAITS=0; CAP_MAX=6
  while [ ! -f "$HUNG_MARKER" ]; do
    CAP_SLEEP="$(python3 - "$CUM" "$RUN_OFF" <<'PY'
import datetime, json, re, sys, time
p, off = sys.argv[1], int(sys.argv[2]); last = None
with open(p, encoding="utf-8", errors="replace") as f:
    f.seek(off)
    for line in f:
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("type") == "result": last = ev
if not last or not last.get("is_error"): sys.exit(0)
txt = str(last.get("result") or "")
if "[1308]" not in txt and "Usage limit reached" not in txt: sys.exit(0)
m = re.search(r"reset at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", txt)
wait = 1800
if m:
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    wait = int(t.timestamp() - time.time()) + 90
print(max(60, min(wait, 6 * 3600)))
PY
)"
    [ -n "$CAP_SLEEP" ] || break
    if [ "$CAP_WAITS" -ge "$CAP_MAX" ]; then
      echo "claude launcher: provider usage cap hit again after $CAP_WAITS wait(s); the turn ends as failed" >&2
      break
    fi
    CAP_WAITS=$((CAP_WAITS+1))
    echo "claude launcher: provider usage cap; waiting ${CAP_SLEEP}s for the stated reset, then resuming the session with this turn's prompt (wait $CAP_WAITS/$CAP_MAX)" >&2
    printf 'wait=%s sleep_s=%s at=%s\n' "$CAP_WAITS" "$CAP_SLEEP" "$(date -u +%FT%TZ)" >> "$WORKDIR/.shim_cap_waits"
    sleep "$CAP_SLEEP"
    SID="$(cat "$CODEX_HOME/claude_last_session_id" 2>/dev/null || true)"
    [ -n "$SID" ] || { echo "claude launcher: no session id to resume after the cap wait; the turn ends as failed" >&2; break; }
    NEWCMD=(); skip=0
    for a in "${CMD[@]}"; do
      if [ "$skip" = 1 ]; then skip=0; continue; fi
      if [ "$a" = "--resume" ]; then skip=1; continue; fi
      NEWCMD+=("$a")
    done
    CMD=("${NEWCMD[@]}" --resume "$SID")
    run_claude_once
  done
  if [ -f "$HUNG_MARKER" ]; then
    n_assist=$(tail -c +$((BASE_OFF+1)) "$CUM" 2>/dev/null | grep -c '"type":"assistant"'); n_assist=${n_assist:-0}
    if [ "$n_assist" -eq 0 ]; then
      echo "claude launcher: idle-killed with no model output: provider outage, the driver retries" >&2
      STATUS=124
    else
      echo "claude launcher: idle-killed after $n_assist assistant events: the model hung, the turn fails" >&2
      STATUS=97
    fi
  fi
fi

# ---- the closing message, for the driver's --output-last-message ---------------------------------
OUT="$(python3 - "$TRACE_SRC" "$TRACE_OFF" <<'PY'
import json, sys
p, off = sys.argv[1], int(sys.argv[2]); final = ""
with open(p, encoding="utf-8", errors="replace") as f:
    f.seek(off)
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "result":
            final = ev.get("result") or final
print(final, end="")
PY
)"
[ -n "$OUTFILE" ] && printf '%s' "$OUT" > "$OUTFILE"
printf '%s\n' "$OUT"
cat shim-claude-stderr.txt >&2 2>/dev/null
rm -f shim-claude-stderr.txt
if [ -n "${CODEX_HOME:-}" ] && [ ! -s "$CODEX_HOME/claude_last_session_id" ]; then   # the first event took over 120 s
  TURN_SID="$(python3 - "$TRACE_SRC" "$TRACE_OFF" <<'PY'
import json, sys
p, off = sys.argv[1], int(sys.argv[2]); sid = ""
with open(p, encoding="utf-8", errors="replace") as f:
    f.seek(off)
    for line in f:
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("session_id"): sid = ev["session_id"]; break
print(sid, end="")
PY
)"
  if [ -n "$TURN_SID" ]; then
    printf '%s\n' "$TURN_SID" > "$CODEX_HOME/claude_last_session_id"
    mkdir -p "$CODEX_HOME/sessions"
    [ -e "$CODEX_HOME/sessions/rollout-$TURN_SID.jsonl" ] || ln "$CODEX_HOME/claude_trace_cum.jsonl" "$CODEX_HOME/sessions/rollout-$TURN_SID.jsonl" 2>/dev/null || true
  fi
fi
exit "$STATUS"
