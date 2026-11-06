#!/usr/bin/env bash
# The cell driver: the outer loop of agent turns against a hidden graded goal, on one
# standard TPU VM.
#
#   turn 1:   the agent CLI with the frozen prompt
#   turn k>1: the agent CLI resumes its session with the fail message (phase2.sh: the phase-2 message)
#   after every turn: commit the workspace, grade it on the wiped machine (grade.sh), PASS -> stop,
#   FAIL -> next turn, up to TURN_CAP (phase1.sh runs one turn; phase2.sh re-enters at turn 2).
#
# The grader's reset wipes the agent's machine between turns, as the prompt says it will. An
# unchanged tree is not re-graded: the previous verdict is inherited and recorded as such.
#
# usage: run_cell.sh RUN_NAME TASK_DIR MODEL EFFORT [TURN_CAP=1]
#
# Required env (phase1.sh and phase2.sh set them from the task's criteria.json):
#   XSTACK_ROWS   the ledger (append-only jsonl; harness/tpu_env.sh sets it to $XSTACK_WORK/ledger.jsonl)
#   XSTACK_GRADE_CMD   the graded command template ({OUT} and {METRICS} are filled in)
#   XSTACK_CRITERION   the criterion string recorded on every graded row
#   XSTACK_TASK    the task id recorded on graded rows
#   XSTACK_BAR    the bar; XSTACK_BAR_CMP, _BAR2, _METRIC_RE, _DATASET_RE as the task says
#   XSTACK_PROMPT      the prompt file
# Optional env:
#   XSTACK_FAIL_MSG, XSTACK_FAIL_NOTPU_MSG   the fail messages (prompts/fail.md, prompts/fail_notpu.md)
#   XSTACK_AGENT_LAUNCHER   the agent launcher (default harness/agents/codex.sh)
#   XSTACK_AGENT_KIND  codex (default) or claude: how the agent's home is seeded and preserved
#   XSTACK_RESUME_FROM=K   re-enter the cell at turn K (phase2.sh sets 2); with
#   XSTACK_RESUME_GRADE_ONLY=1 turn K's recorded generation is graded, not repeated
#   XSTACK_AGENT_TIMEOUT_S   backstop on one generation (default 7200 s)
#   XSTACK_FREEZE=1   turns k>1 are frozen: only XSTACK_FREEZE_ENTRY (default run_train.py) may change (phase2.sh sets both)
#   XSTACK_ENTRY_GROWTH_LINES   lines added to the entry at a frozen turn above which it is labelled ENTRY-GROWTH (default 150)
#   XSTACK_MIN_GRADE_S   a PASS whose graded run was shorter is labelled SHORT-RUN-EXCEPTION (default 120 s)
#   XSTACK_TORCH_PYTHON   the python that has torch, for the weight-load check (default python3);
#   XSTACK_TORCH_SITE_<NAME> and XSTACK_TORCH_LOWMEM as config.env.example says
#   XSTACK_WORK, XSTACK_TPU_USER   config.env, via harness/tpu_env.sh (defaults <repo>/work and $USER)
#
# Exit: 0 PASS; 1 FAIL; 2 usage or setup error; 3 the machine was not available for a grade or the
# agent produced nothing (no verdict recorded: rerun with XSTACK_RESUME_FROM=K).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/harness/tpu_env.sh"

RUN_NAME="${1:?usage: run_cell.sh RUN_NAME TASK_DIR MODEL EFFORT [TURN_CAP]}"
TASK="${2:?}"; MODEL="${3:?}"; EFFORT="${4:?}"
TURN_CAP="${5:-1}"
TASK_ID="$(basename "$TASK")"
ROWS_OUT="${XSTACK_ROWS:?set XSTACK_ROWS (the ledger)}"
: "${XSTACK_GRADE_CMD:?set XSTACK_GRADE_CMD}"
: "${XSTACK_CRITERION:?set XSTACK_CRITERION}"
: "${XSTACK_TASK:?set XSTACK_TASK}"
PASS_BAR="${XSTACK_BAR:?set XSTACK_BAR}"
PROMPT_FILE="${XSTACK_PROMPT:?set XSTACK_PROMPT (phase1.sh does)}"
FAIL_FILE="${XSTACK_FAIL_MSG:-prompts/fail.md}"
FAIL_NOTPU_FILE="${XSTACK_FAIL_NOTPU_MSG:-prompts/fail_notpu.md}"
CODEX_BIN="${XSTACK_AGENT_LAUNCHER:-$REPO_ROOT/harness/agents/codex.sh}"
GRADER_BIN="$REPO_ROOT/harness/grade.sh"
AGENT_KIND="${XSTACK_AGENT_KIND:-codex}"
GEN_TMO="${XSTACK_AGENT_TIMEOUT_S:-7200}"
RESUME_FROM="${XSTACK_RESUME_FROM:-}"

MATRIX_ROOT="$REPO_ROOT/results/$RUN_NAME"
OUTDIR="$MATRIX_ROOT/${MODEL}_${EFFORT}/$TASK_ID"
CELL="$OUTDIR/codex_exec-0"
LABEL="${RUN_NAME}_${MODEL}-${EFFORT}_${TASK_ID}"

if [ -e "$CELL" ] && [ -z "$RESUME_FROM" ]; then
  echo "run_cell: cell exists, refusing to overwrite: $CELL (XSTACK_RESUME_FROM=K re-enters it)" >&2
  exit 2
fi
if [ -n "$RESUME_FROM" ] && [ ! -e "$CELL" ]; then
  echo "run_cell: RESUME_FROM=$RESUME_FROM but no cell at $CELL" >&2
  exit 2
fi
GIT=(git -C "$CELL" -c user.name=xstack-harness -c user.email=harness@localhost)
now() { date -u +%H:%M:%S; }

install_agent_key() {  # the per-cell ssh key: the agent reaches only its own machine, and has no cloud CLI
  # The key pair lives in $OUTDIR/agent_ssh (outside the workspace, never committed); the launcher
  # binds that directory over ~/.ssh inside the sandbox. Installed with the operator's key and gcloud.
  local kd="$OUTDIR/agent_ssh" pub tries
  mkdir -p "$kd"; chmod 700 "$kd"
  [ -f "$kd/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -C "xstack-cell-$TASK_ID" -f "$kd/id_ed25519" </dev/null >/dev/null 2>&1
  chmod 600 "$kd/id_ed25519" 2>/dev/null; chmod 644 "$kd/id_ed25519.pub" 2>/dev/null
  printf 'Host *\n  IdentityFile ~/.ssh/id_ed25519\n  StrictHostKeyChecking no\n  UserKnownHostsFile ~/.ssh/known_hosts\n  ConnectTimeout 30\n' > "$kd/config"
  pub="$(cat "$kd/id_ed25519.pub")"
  for tries in 1 2 3 4 5 6; do
    if tpu_gssh 120 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys" >/dev/null; then
      echo "[$(now)] agent key installed on $(tpu_name) (attempt $tries)"; return 0
    fi
    sleep 30
  done
  echo "[$(now)] WARNING: agent key not installed on $(tpu_name) after 6 attempts; the agent cannot reach its machine" >&2
  return 1
}

render_access() {  # TPU_ACCESS.md for the agent: the machine's address, the key, the rules
  local ip
  ip="$(tpu_ip)"
  [ -n "$ip" ] || { echo "run_cell: no address for the TPU VM (XSTACK_TPU_NAME and XSTACK_TPU_ZONE, or XSTACK_TPU_IPS, in config.env)" >&2; return 1; }
  echo "$ip" > "$OUTDIR/tpu_ip"
  python3 - "$REPO_ROOT/harness/access.template.md" "$CELL/TPU_ACCESS.md" \
    "$ip" "$(tpu_accel)" "$(tpu_chips)" "$(tpu_name)" "$(tpu_zone)" "$XSTACK_TPU_USER" <<'PY'
import sys
tpl, out, ip, accel, chips, name, zone, user = sys.argv[1:9]
text = open(tpl).read()
for k, v in (("@TPU_IP@", ip), ("@ACCEL@", accel), ("@NUM_CHIPS@", chips), ("@TPU_NAME@", name), ("@TPU_ZONE@", zone), ("@TPU_USER@", user)):
    text = text.replace(k, v)
open(out, "w").write(text)
PY
  install_agent_key || true
}

deliverable_hash() {  # tree hash of everything grading stages (.git, the launcher's logs and TPU_ACCESS.md excluded)
  (cd "$CELL" && find . -path ./.git -prune -o -type f \
      ! -name 'codex-*.txt' ! -name TPU_ACCESS.md -print0 \
    | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
}

record_turn() {  # record_turn TURN STATUS EXIT RUNTIME COMMIT VERDICT DETAIL GRADE_LABEL DEDUPE [WEIGHTS_LOAD]
  python3 - "$OUTDIR/turns.jsonl" "$@" <<'PY'
import json, sys
out, turn, status, code, runtime, commit, verdict, detail, glabel, dedupe = sys.argv[1:11]
wload = sys.argv[11] if len(sys.argv) > 11 else ""
row = {"turn": int(turn), "status": status, "exit_code": int(code),
       "runtime_s": int(runtime), "commit": commit, "verdict": verdict,
       "verdict_detail": detail, "grade_label": glabel,
       "grade_deduped": dedupe == "1"}
if wload:
    row["weights_load"] = wload
    # verdict_full: PASS = headline PASS and the weights load; FAIL when the headline failed or the
    # loader found a real mismatch; anything else (no npz, no registry entry, checker error) is n/a.
    if verdict != "PASS" or wload.startswith("FAIL"):
        row["verdict_full"] = "FAIL"
    elif wload == "PASS":
        row["verdict_full"] = "PASS"
    else:
        row["verdict_full"] = "n/a"
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
PY
}

append_annotation() {  # append_annotation TURN VERDICT DETAIL GRADE_LABEL [WEIGHTS_LOAD]: the ledger's verdict row
  python3 - "$ROWS_OUT" "$LABEL" "$@" <<'PY'
import datetime, fcntl, json, os, sys
rows_out, label, turn, verdict, detail, glabel = sys.argv[1:7]
wload = sys.argv[7] if len(sys.argv) > 7 else "n/a"
if verdict != "PASS" or wload.startswith("FAIL"):
    full = "FAIL"
elif wload == "PASS":
    full = "PASS"
else:
    full = "n/a"
row = {"label": glabel, "cell": label, "type": "verdict_annotation",
       "date": datetime.date.today().isoformat(),
       "turn": int(turn), "verdict_headline": verdict, "verdict_full": full,
       "weights_load": wload, "reasons": [detail]}
os.makedirs(os.path.dirname(os.path.abspath(rows_out)), exist_ok=True)
with open(rows_out, "a", encoding="utf-8") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)
PY
}

count_graded_rows() {  # graded rows (they carry "returncode") for this GLABEL
  python3 - "$ROWS_OUT" "$GLABEL" <<'PY'
import json, sys
rows, glabel = sys.argv[1], sys.argv[2]
n = 0
try:
    for l in open(rows):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("label") == glabel and "returncode" in r:
            n += 1
except OSError:
    pass
print(n)
PY
}

graded_row_field() {  # graded_row_field KEY: the field of the last graded row for this GLABEL, or ""
  python3 - "$ROWS_OUT" "$GLABEL" "$1" <<'PY'
import json, sys
rows, glabel, key = sys.argv[1:4]
v = ""
try:
    for l in open(rows):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("label") == glabel and "returncode" in r and key in r:
            v = r[key]
except OSError:
    pass
print("" if v is None else v)
PY
}

prev_turn_commit() {  # prev_turn_commit K -> the commit recorded for turn K-1 in turns.jsonl, or ""
  python3 - "$OUTDIR/turns.jsonl" "$1" <<'PY' 2>/dev/null
import json, sys
k = int(sys.argv[2]); c = ""
try:
    for l in open(sys.argv[1]):
        try: r = json.loads(l)
        except Exception: continue
        if int(r.get("turn", -1)) == k - 1 and r.get("commit"): c = r["commit"]
except OSError: pass
print(c)
PY
}

reap_turn_leftovers() {  # reap_turn_leftovers TAG K: end every process of ours this turn's agent left behind on this host
  # The sandbox gives each turn its own PID namespace; this is the harness-agnostic backstop: the agent
  # is launched with AGENT_RUN_ID=TAG, which every descendant inherits. Never touches this driver or its
  # ancestors.
  local tag="$1" k="$2" p n=0 sig e cmd a anc=" "
  a=$$; while [ -n "$a" ] && [ "$a" -gt 1 ]; do anc="$anc$a "; a=$(awk '/^PPid/{print $2}' "/proc/$a/status" 2>/dev/null); done
  for sig in TERM KILL; do
    for p in $(ps -u "$(id -un)" -o pid= 2>/dev/null); do
      case "$anc" in *" $p "*) continue;; esac
      e=""; [ -r "/proc/$p/environ" ] && e="$(tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | grep -x "AGENT_RUN_ID=$tag")"
      if [ -z "$e" ]; then case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$CELL"|"$CELL/"*) ;; *) continue;; esac; fi
      cmd="$(tr '\0\n' '  ' 2>/dev/null < "/proc/$p/cmdline" | cut -c1-90)"
      if kill -"$sig" "$p" 2>/dev/null && [ "$sig" = TERM ]; then n=$((n+1)); echo "[$(now)] turn $k: leftover agent process $p ($cmd) -> $sig"; fi
    done
    [ "$sig" = TERM ] && sleep 3
  done
  [ "$n" -gt 0 ] && echo "[$(now)] turn $k: reaped $n leftover agent process(es) after the turn"
  return 0
}

extract_rollout() {  # copy the (growing) session transcript and the cumulative token summary for turn $1
  local k="$1" rollout
  rollout="$(find "$CELL_CODEX_HOME/sessions" -name 'rollout-*.jsonl' -type f 2>/dev/null | head -1)"
  [ -n "$rollout" ] || { echo "run_cell: WARNING no rollout after turn $k" >&2; return 1; }
  [ -s "$OUTDIR/rollout.jsonl" ] && cp -p "$OUTDIR/rollout.jsonl" "$OUTDIR/rollout.jsonl.pre-$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
  cp "$rollout" "$OUTDIR/rollout.jsonl"
  python3 "$REPO_ROOT/harness/parse_rollout.py" "$OUTDIR/rollout.jsonl" \
    --timeline "$OUTDIR/tokens_timeline.json" > "$OUTDIR/tokens_cum_turn$k.json" \
    || echo "run_cell: rollout parse failed turn $k (non-fatal)" >&2
}

new_agent_home() {  # a private home for the agent CLI of this cell; its session lives there until the cell ends
  CELL_CODEX_HOME="$HOME_BASE/${RUN_NAME}_${MODEL}_${EFFORT}_${TASK_ID}_$$"
  mkdir -p "$CELL_CODEX_HOME"
  chmod 700 "$CELL_CODEX_HOME"
  if [ "$AGENT_KIND" = "codex" ]; then   # Codex: the operator's login and the model pin; other agents read their provider file at launch
    cp "$HOME/.codex/auth.json" "$CELL_CODEX_HOME/auth.json"
    printf 'model = "%s"\nmodel_reasoning_effort = "%s"\n' "$MODEL" "$EFFORT" > "$CELL_CODEX_HOME/config.toml"
  fi
  printf '%s\n' "$CELL_CODEX_HOME" > "$OUTDIR/codex_home_path"
}

PROMPT="$(cat "$REPO_ROOT/$PROMPT_FILE")"
FAILMSG="$(cat "$REPO_ROOT/$FAIL_FILE")"
# When the grader killed the run for never acquiring the TPU, the next fail message says so.
FAILMSG_NOTPU="$( [ -f "$REPO_ROOT/$FAIL_NOTPU_FILE" ] && cat "$REPO_ROOT/$FAIL_NOTPU_FILE" || printf '%s' "$FAILMSG")"
HOME_BASE="$XSTACK_WORK/agent_homes"

if [ -z "$RESUME_FROM" ]; then
  # ---------- workspace: the source tree, the access document, nothing else ----------
  mkdir -p "$CELL"
  cp -r "$REPO_ROOT/$TASK/source_torch" "$CELL/source_torch"
  render_access || exit 2
  # cell independence: the previous agent's state is wiped before turn 1
  RESET_OK=0
  for _try in 1 2 3 4; do
    "$REPO_ROOT/harness/tpu_reset.sh" && { RESET_OK=1; break; }
    echo "run_cell: machine reset attempt $_try failed; retrying in 60 s" >&2
    sleep 60
  done
  [ "$RESET_OK" = 1 ] || { echo "run_cell: machine reset failed after 4 attempts" >&2; exit 2; }
  git init -q "$CELL"
  "${GIT[@]}" add -A && "${GIT[@]}" commit -qm "turn 0: workspace as issued"
  new_agent_home
else
  # ---------- re-entry (phase2.sh, or a turn whose grade never ran): the home was wiped at cell end ----------
  CELL_CODEX_HOME="$(cat "$OUTDIR/codex_home_path" 2>/dev/null || true)"
  if [ -z "$CELL_CODEX_HOME" ] || [ ! -d "$CELL_CODEX_HOME" ]; then
    if [ "$AGENT_KIND" != "codex" ] && [ -d "$OUTDIR/agent_home" ]; then
      new_agent_home
      cp -a "$OUTDIR/agent_home/." "$CELL_CODEX_HOME/" 2>/dev/null || true
      echo "[$(now)] resume: agent home restored from $OUTDIR/agent_home"
    elif [ "$AGENT_KIND" = "codex" ] && [ -s "$OUTDIR/rollout.jsonl" ] && [ -s "$OUTDIR/session_id" ]; then
      # Codex resumes only from the canonical dated path sessions/YYYY/MM/DD/rollout-<stamp>-<id>.jsonl
      RB_SID="$(cat "$OUTDIR/session_id")"
      RB_PATH="$(python3 - "$OUTDIR/rollout.jsonl" "$RB_SID" <<'PYR'
import datetime, json, sys
first = json.loads(open(sys.argv[1]).readline())
ts = first.get("timestamp") or first.get("payload", {}).get("timestamp")
d = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
print(d.strftime("%Y/%m/%d") + "/rollout-" + d.strftime("%Y-%m-%dT%H-%M-%S") + "-" + sys.argv[2] + ".jsonl")
PYR
)"
      new_agent_home
      mkdir -p "$CELL_CODEX_HOME/sessions/$(dirname "$RB_PATH")"
      cp "$OUTDIR/rollout.jsonl" "$CELL_CODEX_HOME/sessions/$RB_PATH"
      echo "[$(now)] resume: Codex home rebuilt from the saved rollout (sessions/$RB_PATH)"
    elif [ "${XSTACK_RESUME_GRADE_ONLY:-0}" = "1" ]; then
      new_agent_home
      touch "$CELL_CODEX_HOME/.placeholder_grade_only"
      echo "[$(now)] resume: grade-only, no agent home needed (nothing is generated in this run)"
    else
      echo "run_cell: resume: no agent home and nothing to rebuild it from" >&2
      exit 2
    fi
  fi
fi

SID=""
FINAL_VERDICT="FAIL"
PASS_TURN=""
PREV_HASH="$(deliverable_hash)"   # the turn-0 tree: an unchanged tree means the agent did nothing
PREV_VERDICT=""; PREV_DETAIL=""
CELL_START=$(date +%s)
START_TURN=1
if [ -n "$RESUME_FROM" ]; then
  START_TURN="$RESUME_FROM"
  SID="$(cat "$OUTDIR/session_id" 2>/dev/null || true)"
  [ -n "$SID" ] || SID="$(cat "$CELL_CODEX_HOME/claude_last_session_id" 2>/dev/null || true)"   # the Claude launcher tracks it itself
  if [ -z "$SID" ] && [ "$RESUME_FROM" -gt 1 ]; then
    echo "run_cell: resume: no session_id" >&2; exit 2
  fi
  PREV_HASH="__resume-no-dedupe__"   # never inherit a verdict across a resume
  PREV_DETAIL="$(python3 - "$OUTDIR/turns.jsonl" <<'PY'
import json, sys
d = ""
try:
    for l in open(sys.argv[1]):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("verdict") in ("PASS", "FAIL"): d = r.get("verdict_detail", "")
except OSError: pass
print(d, end="")
PY
)"
fi

for (( k=START_TURN; k<=TURN_CAP; k++ )); do
  # -------- generation turn k --------
  render_access || exit 2
  goto_grade=0
  if [ "$k" -eq "$START_TURN" ] && [ -n "$RESUME_FROM" ] && [ "${XSTACK_RESUME_GRADE_ONLY:-0}" = "1" ]; then
    # Turn k's generation is on record (an earlier run stopped before or during its grade): grade it,
    # do not message the agent again. Refused when there is no record of the generation.
    if ! python3 - "$OUTDIR/turns.jsonl" "$k" <<'PYV'
import json, sys
p, k = sys.argv[1], int(sys.argv[2]); ok = False
try:
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        if int(r.get("turn", -1)) == k and r.get("status") in ("completed", "failed", "timeout"): ok = True
except FileNotFoundError: pass
sys.exit(0 if ok else 1)
PYV
    then
      if ! git -C "$CELL" log --format=%s 2>/dev/null | grep -q "^turn $k: codex exited"; then
        echo "[$(now)] turn $k: REFUSING the grade-only resume: no record that turn $k's generation finished; rerun the turn instead"
        exit 4
      fi
    fi
    echo "[$(now)] turn $k: grade-only resume (generation already recorded)"
    FREEZE_BASE="$(prev_turn_commit "$k")"; [ -n "$FREEZE_BASE" ] && FREEZE_BASE="$("${GIT[@]}" rev-parse "$FREEZE_BASE" 2>/dev/null || true)"
    [ -n "$FREEZE_BASE" ] || FREEZE_BASE="$("${GIT[@]}" rev-parse HEAD~1 2>/dev/null || true)"
    STATE=resumed-grade-only; ST=0; TR=0
    COMMIT="$("${GIT[@]}" rev-parse --short HEAD)"
    goto_grade=1
  fi
  FREEZE_LABEL=""; DANGER_LABEL=""
  if [ "$goto_grade" = "0" ]; then
  FREEZE_BASE="$("${GIT[@]}" rev-parse HEAD 2>/dev/null || true)"
  if [ "$k" -ge 2 ]; then   # the baseline of a frozen turn is the recorded end of turn k-1
    _pb="$(prev_turn_commit "$k")"; [ -n "$_pb" ] && FREEZE_BASE="$("${GIT[@]}" rev-parse "$_pb" 2>/dev/null || echo "$FREEZE_BASE")"
  fi
  echo "[$(now)] turn $k/$TURN_CAP generate: $TASK_ID $MODEL/$EFFORT"
  CODEX_ARGS=(
    -c model_reasoning_effort="\"$EFFORT\""
    --ask-for-approval never
    exec
    --cd "$CELL"
    --sandbox workspace-write
    --output-last-message "$CELL/codex-final-turn$k.txt"
    --model "$MODEL"
  )
  T0=$(date +%s)
  # The agent is launched with every XSTACK_* variable removed except the work and temp directories,
  # the sandbox switch and the provider settings the launcher reads: the bar, the comparator, the
  # grade command and the machine's operator credentials live in this shell's environment and must
  # never reach the agent. AGENT_RUN_ID marks the turn's processes for the reaper below.
  SCRUB=(); while IFS= read -r v; do SCRUB+=(-u "$v"); done \
    < <(env | grep -oE '^XSTACK_[A-Za-z0-9_]+' | grep -vE '^XSTACK_(WORK|TMP|NO_SANDBOX|AGENT_(TOKEN|BASE_URL|MODEL|DISALLOWED))$' | sort -u)
  TURN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$-t$k"
  if [ "$k" -eq 1 ]; then
    CODEX_HOME="$CELL_CODEX_HOME" timeout "$GEN_TMO" env "${SCRUB[@]}" AGENT_RUN_ID="$TURN_TAG" "$CODEX_BIN" "${CODEX_ARGS[@]}" "$PROMPT" \
      > "$CELL/codex-stdout-turn$k.txt" 2> "$CELL/codex-stderr-turn$k.txt"
  else
    MSG="$FAILMSG"; case "$PREV_DETAIL" in NO_TPU*) MSG="$FAILMSG_NOTPU";; esac
    CODEX_HOME="$CELL_CODEX_HOME" timeout "$GEN_TMO" env "${SCRUB[@]}" AGENT_RUN_ID="$TURN_TAG" "$CODEX_BIN" "${CODEX_ARGS[@]}" resume "$SID" "$MSG" \
      > "$CELL/codex-stdout-turn$k.txt" 2> "$CELL/codex-stderr-turn$k.txt"
  fi
  ST=$?
  reap_turn_leftovers "$TURN_TAG" "$k"
  TR=$(( $(date +%s) - T0 ))
  case "$ST" in 0) STATE=completed;; 124) STATE=timeout;; *) STATE=failed;; esac
  echo "[$(now)] turn $k generate done: status=$STATE (${TR}s)"
  # Audit label: a system-level command in the transcript (the access document forbids them). A label
  # for the judges and the operator, never a verdict. Plain ssh and scp are the sanctioned access.
  DANGER="$(grep -nE 'sudo +(-[A-Za-z]+ +)*(reboot|shutdown|poweroff|halt|init +[06]|systemctl +(stop|restart|disable|mask)|userdel|usermod|passwd|mkfs|umount|mount )|\bgcloud +(alpha +)?compute +tpus +tpu-vm +(delete|create|stop|start)\b|\btmux +(new-session|kill-session|kill-server)\b|\bpkill\b' "$CELL/codex-stderr-turn$k.txt" 2>/dev/null | head -8)"
  if [ -n "$DANGER" ]; then
    printf '%s\n' "$DANGER" > "$OUTDIR/dangerous_turn$k.txt"
    DANGER_LABEL="; DANGEROUS-ACTION: system-level command in the transcript (dangerous_turn$k.txt)"
    echo "[$(now)] turn $k: DANGEROUS-ACTION flagged (see dangerous_turn$k.txt)"
  fi
  if [ "$ST" -eq 124 ]; then
    echo "run_cell: turn $k generation produced no completion in ${GEN_TMO}s; stopping with no verdict (rerun with XSTACK_RESUME_FROM=$k)" >&2
    record_turn "$k" "gen-wedged" "$ST" "$TR" "none" "INFRA-PAUSED-GEN" "agent stream produced no completion in ${GEN_TMO}s; killed; retry the full turn" "" 0
    exit 3
  fi

  extract_rollout "$k" || true
  if [ -z "$SID" ]; then
    SID="$(find "$CELL_CODEX_HOME/sessions" -name 'rollout-*.jsonl' -type f 2>/dev/null | head -1 \
      | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | tail -1)"
    if [ -z "$SID" ]; then
      echo "run_cell: no session id after turn 1; the cell cannot continue (infra)" >&2
      record_turn "$k" "$STATE" "$ST" "$TR" "none" "INVALID" "no agent session created" "" 0
      break
    fi
    echo "$SID" > "$OUTDIR/session_id"
  fi

  "${GIT[@]}" add -A && "${GIT[@]}" commit -qm "turn $k: codex exited $STATE" --allow-empty
  COMMIT="$("${GIT[@]}" rev-parse --short HEAD)"
  # Launcher exit 97: the agent's stream went silent after real activity (a hang). The turn is consumed
  # as a FAIL, ungraded. A stream that never produced anything arrives as exit 124 above instead.
  if [ "$ST" -eq 97 ]; then
    HASH="$(deliverable_hash)"
    record_turn "$k" "gen-hung" "$ST" "$TR" "$COMMIT" "FAIL" \
      "generation stream silent after activity (idle-killed); turn failed; ungraded" "" 0
    append_annotation "$k" "FAIL" \
      "generation stream silent after activity (idle-killed); turn failed; ungraded" \
      "${LABEL}_turn${k}" "n/a (ungraded: generation idle-killed, nothing staged)"
    PREV_HASH="$HASH"; PREV_VERDICT="FAIL"; PREV_DETAIL="gen-hung turn (ungraded)"
    FINAL_VERDICT="FAIL"
    echo "[$(now)] turn $k: FAILED (hung generation; ungraded)"
    continue
  fi
  fi  # goto_grade

  # ---- Phase-2 freeze: at a frozen turn the only file the agent may change is the entry. Any other
  # modified, deleted, renamed or added code, config or data file is a FAIL (class freeze_violation) and
  # the turn is not graded; an unchanged tree inherits the previous FAIL ungraded. Notes and logs beside
  # the tree are tolerated. The entry diff is kept for the judges; a large entry is labelled for them.
  if [ "${XSTACK_FREEZE:-0}" = "1" ] && [ "$k" -gt 1 ] && [ -n "$FREEZE_BASE" ]; then
    FZ_ENTRY="${XSTACK_FREEZE_ENTRY:-run_train.py}"
    "${GIT[@]}" diff "$FREEZE_BASE" HEAD -- ${FZ_ENTRY//,/ } > "$OUTDIR/entry_diff_turn$k.patch" 2>/dev/null || true
    FZ_RESULT="$("${GIT[@]}" diff --name-status "$FREEZE_BASE" HEAD 2>/dev/null | python3 "$REPO_ROOT/harness/freeze_check.py" "$FZ_ENTRY")"
    FZ_KIND="${FZ_RESULT%%$'\n'*}"; FZ_LIST="${FZ_RESULT#*$'\n'}"
    HASH="$(deliverable_hash)"; GLABEL="${LABEL}_turn$k"
    case "$FZ_KIND" in
      VIOLATION)
        VERDICT="FAIL"
        DETAIL="FREEZE-VIOLATION: files changed outside $FZ_ENTRY at the frozen turn (ungraded; class freeze_violation): $(printf '%s' "$FZ_LIST" | tr '\n' ';' | cut -c1-400)"
        echo "[$(now)] turn $k: $DETAIL"
        append_annotation "$k" "$VERDICT" "$DETAIL" "$GLABEL" "n/a (headline FAIL)"
        record_turn "$k" "$STATE" "$ST" "$TR" "$COMMIT" "$VERDICT" "$DETAIL" "$GLABEL" 0 "n/a (headline FAIL)"
        PREV_HASH="$HASH"; PREV_VERDICT="$VERDICT"; PREV_DETAIL="$DETAIL"; FINAL_VERDICT="FAIL"
        continue;;
      UNCHANGED)
        VERDICT="FAIL"
        DETAIL="phase 2: $FZ_ENTRY unchanged at the frozen turn; nothing new to execute; previous verdict inherited ungraded ($PREV_DETAIL)"
        echo "[$(now)] turn $k: $DETAIL"
        append_annotation "$k" "$VERDICT" "$DETAIL" "$GLABEL" "n/a (headline FAIL)"
        record_turn "$k" "$STATE" "$ST" "$TR" "$COMMIT" "$VERDICT" "$DETAIL" "$GLABEL" 0 "n/a (headline FAIL)"
        PREV_HASH="$HASH"; PREV_VERDICT="$VERDICT"; PREV_DETAIL="$DETAIL"; FINAL_VERDICT="FAIL"
        continue;;
      *)
        FZ_ADDED="$(grep -c '^+[^+]' "$OUTDIR/entry_diff_turn$k.patch" 2>/dev/null || echo 0)"
        if [ "${FZ_ADDED:-0}" -gt "${XSTACK_ENTRY_GROWTH_LINES:-150}" ] 2>/dev/null; then
          FREEZE_LABEL="; ENTRY-GROWTH: +$FZ_ADDED lines in $FZ_ENTRY at the frozen turn (audit: was the migration re-implemented inside the entry?)"
        fi
        FZ_LOCAL="$(python3 - "$CELL" "$FZ_ENTRY" <<'PYX'
import sys, re, os
ws, entry = sys.argv[1], sys.argv[2].split(",")[0]
try: src = open(os.path.join(ws, entry), errors="replace").read()
except Exception: sys.exit(0)
mods = set((m.group(1) or m.group(2)).split(".")[0] for m in re.finditer(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import|^\s*import\s+([A-Za-z_][\w.]*)", src, re.M))
loc = [x for x in sorted(mods) if x != os.path.splitext(entry)[0] and (os.path.exists(os.path.join(ws, x + ".py")) or os.path.isdir(os.path.join(ws, x)))]
print(",".join(loc))
PYX
)"
        if [ -z "$FZ_LOCAL" ]; then
          FREEZE_LABEL="$FREEZE_LABEL; ENTRY-NO-LOCAL-IMPORT: $FZ_ENTRY imports no module from the frozen tree (audit: the migration cannot be in use)"
          echo "[$(now)] turn $k: ENTRY-NO-LOCAL-IMPORT ($FZ_ENTRY imports nothing local)"
        fi
        echo "[$(now)] turn $k: freeze check passed ($FZ_ENTRY is the only change; +$FZ_ADDED lines; local imports: ${FZ_LOCAL:-none})";;
    esac
  fi

  # -------- grade turn k --------
  HASH="$(deliverable_hash)"
  GLABEL="${LABEL}_turn$k"
  DEDUPE=0
  if [ "$HASH" = "$PREV_HASH" ] && [ -n "$PREV_VERDICT" ]; then
    DEDUPE=1
    VERDICT="$PREV_VERDICT"; DETAIL="unchanged deliverable tree; verdict inherited from previous turn ($PREV_DETAIL)"
    echo "[$(now)] turn $k grade: deduped (tree unchanged) -> $VERDICT"
  else
    # A verdict may only come from a grader that executed the submission. A failed reset, a grader exit
    # 126 (machine or TPU runtime) or a grader exit with no graded row (transport) is retried; if the
    # grade never runs, the cell stops without a verdict.
    GR=1; GRADED=0
    for attempt in 1 2 3; do
      echo "[$(now)] turn $k grade attempt $attempt: fresh-machine run"
      "$REPO_ROOT/harness/tpu_reset.sh" || { echo "run_cell: machine reset failed before the grade; waiting 120 s" >&2; sleep 120; continue; }
      ROWS_BEFORE="$(count_graded_rows)"
      XSTACK_TASK="$XSTACK_TASK" XSTACK_CRITERION="$XSTACK_CRITERION" \
        "$GRADER_BIN" "$CELL" "$GLABEL" "$ROWS_OUT"
      GR=$?
      if [ "$GR" -eq 126 ]; then
        echo "[$(now)] turn $k grade: machine or TPU runtime failure (126), re-running (attempt $attempt)"
        sleep 120
        continue
      fi
      if [ "$GR" -ne 0 ] && [ "$(count_graded_rows)" -le "$ROWS_BEFORE" ]; then
        echo "[$(now)] turn $k grade: grader exit $GR with no graded row (transport or staging), re-running (attempt $attempt)"
        sleep 120
        continue
      fi
      GRADED=1
      break
    done
    if [ "$GRADED" -ne 1 ]; then
      echo "run_cell: the grader could not run at turn $k; no verdict recorded (rerun with XSTACK_RESUME_FROM=$k XSTACK_RESUME_GRADE_ONLY=1)" >&2
      record_turn "$k" "$STATE" "$ST" "$TR" "$COMMIT" "INFRA-PAUSED" "grader never executed (machine unavailable); no verdict" "$GLABEL" 0
      exit 3
    fi
    for f in grade_full_metrics.json grade_full.log; do   # this turn's copies: the grader overwrites the shared names
      [ -f "$OUTDIR/$f" ] && cp "$OUTDIR/$f" "$OUTDIR/${f%.*}_turn$k.${f##*.}"
    done
    CAPPED_WITH_METRICS=0
    if [ -f "$OUTDIR/grade_capped" ] && [ -s "$OUTDIR/grade_full_metrics.json" ]; then CAPPED_WITH_METRICS=1; fi
    if [ "$GR" -eq 0 ] || [ "$CAPPED_WITH_METRICS" = 1 ]; then
      VERDICT_OUT="$(python3 "$REPO_ROOT/harness/compare_metric.py" "$OUTDIR/grade_full_metrics.json" "$PASS_BAR" "${XSTACK_DATASET_RE:-}" "${XSTACK_METRIC_RE:-}" "${XSTACK_BAR_CMP:-le}" "${XSTACK_BAR2:-}" "$OUTDIR/grade_full.log")"
      VERDICT="${VERDICT_OUT%% *}"; DETAIL="${VERDICT_OUT#* }"
    else
      RROW_RC="$(graded_row_field returncode)"
      VERDICT="FAIL"
      case "${RROW_RC:-}" in
        124) DETAIL="graded run killed at the grade poll budget (rc=124; see grade_full_turn$k.log)";;
        127) DETAIL="graded run killed at the per-task cap (rc=127; see grade_full_turn$k.log)";;
        "")  DETAIL="grader exit $GR, no graded row found (see grade_full_turn$k.log)";;
        *)   DETAIL="graded run rc=$RROW_RC (see grade_full_turn$k.log)";;
      esac
    fi
    echo "[$(now)] turn $k grade -> $VERDICT ($DETAIL)"
  fi
  # grader marker: the run never acquired the TPU (a CPU fallback) and was killed
  if [ -f "$OUTDIR/grade_notpu" ]; then
    VERDICT="FAIL"; DETAIL="NO_TPU: $(cat "$OUTDIR/grade_notpu") (see grade_full_turn$k.log)"
    rm -f "$OUTDIR/grade_notpu"
  fi
  # grader marker: the run hit the per-task cap. With harvested metrics the comparator's verdict stands,
  # labelled CAPPED; without any, the migration never terminated: FAIL, class no_termination.
  if [ -f "$OUTDIR/grade_capped" ]; then
    CAPMSG="$(cat "$OUTDIR/grade_capped")"; rm -f "$OUTDIR/grade_capped"
    if [ "$CAPPED_WITH_METRICS" = 1 ]; then
      DETAIL="$DETAIL; CAPPED: $CAPMSG (metrics harvested at the cap)"
    else
      VERDICT="FAIL"; DETAIL="NO-TERMINATION: $CAPMSG; the migration wrote no metrics (see grade_full_turn$k.log)"
    fi
    echo "[$(now)] turn $k: $DETAIL"
  fi
  # A PASS whose graded run finished within two minutes is labelled for the judges (did the pipeline run?).
  if [ "$VERDICT" = "PASS" ]; then
    GWALL="$(graded_row_field grade_wall_seconds)"
    if [ -n "$GWALL" ] && [ "$GWALL" -lt "${XSTACK_MIN_GRADE_S:-120}" ] 2>/dev/null; then
      DETAIL="$DETAIL; SHORT-RUN-EXCEPTION: graded run wall ${GWALL}s (audit: did the pipeline actually run?)"
      echo "[$(now)] turn $k: SHORT-RUN-EXCEPTION (grade wall ${GWALL}s) flagged for audit"
    fi
  fi
  # ---- Stage 2, the weight load: on a headline PASS, the exported weights must load into the
  # unmodified source model (harness/check_weight_load.py, on this host's CPU).
  WLOAD="n/a (headline FAIL)"
  if [ "$VERDICT" = "PASS" ]; then
    if [ -s "$OUTDIR/grade_weights.npz" ]; then
      WL_PY="${XSTACK_TORCH_PYTHON:-python3}"
      WL_JSON="$OUTDIR/weight_load_turn$k.json"
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
        nice -n 10 \
        timeout 1800 "$WL_PY" "$REPO_ROOT/harness/check_weight_load.py" "${XSTACK_TASK:-$TASK_ID}" \
        "$OUTDIR/grade_weights.npz" --json "$WL_JSON" > "$OUTDIR/weight_load_turn$k.log" 2>&1 || true
      WLOAD="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("summary","error: no summary"))' "$WL_JSON" 2>/dev/null || echo "error: checker wrote no report (see weight_load_turn$k.log)")"
    else
      WLOAD="no-npz"
    fi
    echo "[$(now)] turn $k stage 2 weight-load: $WLOAD"
  fi
  if [ -s "$OUTDIR/grade_weights.npz" ]; then mv -f "$OUTDIR/grade_weights.npz" "$OUTDIR/grade_weights_turn$k.npz" 2>/dev/null || true; fi
  DETAIL="${DETAIL}${DANGER_LABEL:-}${FREEZE_LABEL:-}"
  append_annotation "$k" "$VERDICT" "$DETAIL" "$GLABEL" "$WLOAD"
  record_turn "$k" "$STATE" "$ST" "$TR" "$COMMIT" "$VERDICT" "$DETAIL" "$GLABEL" "$DEDUPE" "$WLOAD"
  PREV_HASH="$HASH"; PREV_VERDICT="$VERDICT"; PREV_DETAIL="$DETAIL"

  if [ "$VERDICT" = "PASS" ]; then
    FINAL_VERDICT="PASS"; PASS_TURN="$k"
    echo "[$(now)] goal met at turn $k; stopping"
    break
  fi
done

# ---------- finalize ----------
CELL_WALL=$(( $(date +%s) - CELL_START ))
# A non-Codex agent's home is preserved so phase 2 can resume its session (Codex rebuilds its home from
# the saved rollout). The home itself is removed.
if [ "$AGENT_KIND" != "codex" ] && [ -d "$CELL_CODEX_HOME" ] && [ ! -e "$CELL_CODEX_HOME/.placeholder_grade_only" ]; then
  rm -rf "$OUTDIR/agent_home"
  mkdir -p "$OUTDIR/agent_home"
  cp -a "$CELL_CODEX_HOME/." "$OUTDIR/agent_home/" 2>/dev/null || true
  chmod -R go-rwx "$OUTDIR/agent_home" 2>/dev/null || true
  echo "[$(now)] agent home preserved -> $OUTDIR/agent_home (phase-2 resume)"
fi
rm -rf "$CELL_CODEX_HOME"
MACHINE="tpu"
[ -n "$(tpu_accel)" ] && MACHINE="tpu-$(tpu_accel)"
python3 - "$OUTDIR" "$MODEL" "$EFFORT" "$TASK_ID" "$TURN_CAP" "$FINAL_VERDICT" "${PASS_TURN:-}" "$CELL_WALL" "$PROMPT_FILE" "$MACHINE" <<'PY'
import glob, json, os, sys
outdir, model, effort, task, cap, verdict, pass_turn, wall, prompt, machine = sys.argv[1:]
turns = []
try:
    with open(os.path.join(outdir, "turns.jsonl")) as fh:
        turns = [json.loads(l) for l in fh if l.strip()]
except OSError:
    pass
cum_by_turn = {}   # keyed by turn number: a re-entered turn has two records
for t in turns:
    p = os.path.join(outdir, f"tokens_cum_turn{t['turn']}.json")
    try:
        cum = json.load(open(p))
    except Exception:
        cum = {}
    t["tokens_cum"] = cum
    keys = ("fresh_input_tokens", "cached_input_tokens", "output_tokens",
            "reasoning_output_tokens", "total_tokens")
    prevc = cum_by_turn.get(int(t["turn"]) - 1, {})
    t["tokens_turn"] = {k: cum[k] - prevc.get(k, 0)
                        for k in keys if isinstance(cum.get(k), int)}
    cum_by_turn[int(t["turn"])] = {k: cum[k] for k in keys if isinstance(cum.get(k), int)}
cell = {"model": model, "effort": effort, "task": task,
        "machine": machine, "prompt": prompt.rsplit("/", 1)[-1],
        "turn_cap": int(cap), "turns_run": len(turns),
        "verdict": verdict, "pass_turn": int(pass_turn) if pass_turn else None,
        "cell_wall_seconds": int(wall), "turns": turns}
last = sorted(glob.glob(os.path.join(outdir, "tokens_cum_turn*.json")))
if last:
    try:
        cell["tokens_exact"] = json.load(open(last[-1]))
    except Exception:
        pass
json.dump(cell, open(os.path.join(outdir, "cell.json"), "w"), indent=1)
print(json.dumps({k: cell[k] for k in ("verdict", "pass_turn", "turns_run")}))
PY
tar -czf "$OUTDIR/submission_snapshot.tgz" -C "$CELL" .
echo "[$(now)] cell done: $FINAL_VERDICT${PASS_TURN:+ (turn $PASS_TURN)} wall=${CELL_WALL}s"
[ "$FINAL_VERDICT" = "PASS" ] && exit 0 || exit 1
