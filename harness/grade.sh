#!/usr/bin/env bash
# Grade one submission on the wiped TPU VM: run its setup_env.sh and then the task's grade command at
# the source's full scale, record what happens (exit code, wall clock, the metrics file), pull the
# metrics file, the run log and the exported weights, and append one graded row to the ledger. The
# verdict is the driver's (compare_metric.py on the metrics file); this script runs and records.
#
# usage: grade.sh WORKSPACE LABEL [ROWS_OUT]
#   env: XSTACK_GRADE_CMD    the command template; {OUT} and {METRICS} are filled in
#        XSTACK_TASK     the task id (its criteria.json gives the expected wall)
#        XSTACK_CRITERION    recorded on the graded row
#        XSTACK_GRADE_CAP_S  cap on the graded run (default 10x the task's expected_grade_wall_h, at least 2 h)
#        XSTACK_NOTPU_GRACE_S  seconds a run may live without the TPU before it counts as a CPU fallback (7200)
#        XSTACK_ROWS         the ledger appended to when ROWS_OUT is not given (default $XSTACK_WORK/ledger.jsonl)
#        XSTACK_KEEP_GRADE_DIR=1    keep the grade directory on the machine afterwards
# Exit: 0 the run exited 0; 1 it failed, was capped or killed; 2 nothing ran (missing deliverable,
# staging failed); 126 the machine or the TPU runtime failed, no verdict (the driver reruns the grade).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/harness/tpu_env.sh"

WS="${1:?usage: grade.sh WORKSPACE LABEL [ROWS_OUT]}"
LABEL="${2:?}"
ROWS_OUT="${3:-${XSTACK_ROWS:-$XSTACK_WORK/ledger.jsonl}}"
OUT="$(dirname "$WS")"                      # the cell directory: the grade's files land beside the workspace
POLL_BUDGET_S=259200   # 72 h backstop; the per-task cap is 10x expected_grade_wall_h
TPU="$(tpu_name)"
[ -n "$TPU" ] || { echo "grade: XSTACK_TPU_NAME is not set (config.env)" >&2; exit 2; }
MACHINE="tpu"
[ -n "$(tpu_accel)" ] && MACHINE="tpu-$(tpu_accel)"
echo "[grade] $TPU: one host, transport gcloud"

emit_row() {  # emit_row RC NOTE METRICS_JSON_PATH WALL
  python3 - "$ROWS_OUT" "$LABEL" "$1" "$2" "${3:-}" "$4" "$MACHINE" <<'PY'
import fcntl, json, os, sys
rows_out, label, rc, note, mpath, wall, machine = sys.argv[1:]
row = {"label": label, "repo": os.environ.get("XSTACK_TASK", ""), "machine": machine,
       "criterion": os.environ.get("XSTACK_CRITERION", "UNSET"),
       "descriptive": True, "returncode": int(rc)}
if wall:
    row["grade_wall_seconds"] = int(wall)
try:
    row["metrics"] = json.load(open(mpath))
except Exception:
    pass
if note:
    row["note"] = note
os.makedirs(os.path.dirname(os.path.abspath(rows_out)), exist_ok=True)
with open(rows_out, "a", encoding="utf-8") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    fcntl.flock(fh, fcntl.LOCK_UN)
print(json.dumps(row))
PY
}

for f in setup_env.sh run_train.py; do
  if [ ! -f "$WS/$f" ]; then
    emit_row 2 "missing $f (give-up: recorded as a failure)" "" ""
    exit 2
  fi
done

# ---- stage the submission as one archive (tens of thousands of small files copy in seconds this way)
GD="xstack_grade_$(date -u +%s)"
GDP="~"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/submission"
cp -r "$WS/." "$STAGE/submission/"
rm -f "$STAGE/submission/"codex-*.txt 2>/dev/null
rm -rf "$STAGE/submission/.git" 2>/dev/null   # the turn history never ships to the graded machine
tar cf "$STAGE/submission.tar" -C "$STAGE" submission || { echo "grade: could not archive the submission" >&2; exit 2; }
A_MB=$(( $(stat -c %s "$STAGE/submission.tar") / 1048576 )); A_TMO=$(( 420 + A_MB / 10 ))
echo "[grade] submission archive: ${A_MB} MB (staging timeout ${A_TMO}s)"

# The graded command: {OUT} and {METRICS} are relative to the submission directory on the machine.
RCMD="$(python3 -c "import sys; print(sys.argv[1].replace('{OUT}','../out/ck').replace('{METRICS}','../out/metrics.json'))" "$XSTACK_GRADE_CMD")"
echo "[grade] staging and launching: $RCMD"
tpu_gssh 180 "rm -rf $GDP/xstack_grade_* ; mkdir -p $GDP/$GD/out" || { echo "grade: mkdir on the machine failed" >&2; exit 2; }
tpu_gscp_to "$STAGE/submission.tar" "$GDP/$GD/" "$A_TMO" || { echo "grade: staging failed" >&2; exit 2; }
tpu_gssh 600 "tar xf $GDP/$GD/submission.tar -C $GDP/$GD/ && rm -f $GDP/$GD/submission.tar" || { echo "grade: unpack failed" >&2; exit 2; }
# Detached, with a hermetic envelope: caches, temp and the user site live inside the grade directory and
# die with it. stdin is closed inside nohup so the ssh session returns at once.
tpu_gssh 120 "bash --noprofile --norc -c 'cd $GDP/$GD/submission && nohup bash -c \"mkdir -p $GDP/$GD/hf $GDP/$GD/tmp $GDP/$GD/xdg $GDP/$GD/userbase; export PYTHONUNBUFFERED=1 HF_HOME=$GDP/$GD/hf HF_DATASETS_CACHE=$GDP/$GD/hf/datasets TMPDIR=$GDP/$GD/tmp XDG_CACHE_HOME=$GDP/$GD/xdg PYTHONUSERBASE=$GDP/$GD/userbase PATH=$GDP/$GD/userbase/bin:\\\$PATH; ( source setup_env.sh >> $GDP/$GD/out/grade.log 2>&1 && $RCMD >> $GDP/$GD/out/grade.log 2>&1 ); echo \\\$? > $GDP/$GD/out/rc\" < /dev/null > /dev/null 2>&1 & echo LAUNCHED'" || true
sleep 10
probe="$(tpu_gssh 90 "pgrep -f 'setup_en[v].sh|run_trai[n].py' >/dev/null && echo RUN || { test -f $GDP/$GD/out/rc && echo DONE || echo DEAD; }")"
echo "[grade] launch probe: ${probe:-noreply}"
if [ "${probe:-}" = "DEAD" ]; then echo "grade: the run did not start" >&2; exit 2; fi

START=$(date +%s)
# ---- the per-task cap: 10x the task's expected graded-run wall, at least 2 h. A run that reaches it is
# killed and harvested: the driver grades a harvested metrics file normally (labelled CAPPED) and fails
# a run that wrote none (class no_termination).
CAP_S="${XSTACK_GRADE_CAP_S:-}"
if [ -z "$CAP_S" ]; then
  CAP_S="$(python3 - "$REPO_ROOT/tasks/${XSTACK_TASK:-}/criteria.json" <<'PY'
import json, os, sys
h = None
try:
    h = json.load(open(sys.argv[1])).get("expected_grade_wall_h")
except Exception:
    pass
print(int(max(7200, 10 * float(h) * 3600)) if h else 0)
PY
)"
fi
[ "${CAP_S:-0}" -gt 0 ] 2>/dev/null && echo "[grade] cap ${CAP_S}s (10x the expected wall, at least 2 h) for task ${XSTACK_TASK:-?}"
MOUT="$OUT/grade_full_metrics.json"
LOUT="$OUT/grade_full.log"

kill_run() { tpu_gssh 90 "pkill -9 -f 'run_trai[n].py'; sleep 1; pkill -9 -u \$(id -un) -f 'pytho[n]' 2>/dev/null; true" || true; }
fetch_outputs() {  # the run log and the metrics file, by scp with a cat-over-ssh fallback
  : > "$MOUT"; : > "$LOUT"
  for _try in 1 2 3; do tpu_gscp_from "$GDP/$GD/out/metrics.json" "$MOUT" && break; sleep 10; done
  for _try in 1 2 3; do tpu_gscp_from "$GDP/$GD/out/grade.log" "$LOUT" && break; sleep 10; done
  [ -s "$LOUT" ] || tpu_gssh 300 "cat $GDP/$GD/out/grade.log 2>/dev/null" > "$LOUT" || true
  [ -s "$MOUT" ] || tpu_gssh 120 "cat $GDP/$GD/out/metrics.json 2>/dev/null" > "$MOUT" || true
}
npz_export_pull() {  # run the submission's export_torch.py on the machine and pull the weights (stage 2)
  rm -f "$OUT/grade_weights.npz" "$OUT/grade_export.log" "$OUT/grade_export.rc" "$OUT/grade_pull.log"
  tpu_gssh 2400 "bash --noprofile --norc -c 'cd $GDP/$GD/submission && { source setup_env.sh >/dev/null 2>&1 || true; python export_torch.py --ckpt $GDP/$GD/out/ck --out $GDP/$GD/out/weights.npz; } > $GDP/$GD/out/export.log 2>&1; echo \\$? > $GDP/$GD/out/export.rc; ls -la $GDP/$GD/out/weights.npz >> $GDP/$GD/out/export.log 2>&1; cat $GDP/$GD/out/export.rc'" > "$OUT/grade_export.rc" 2>/dev/null || true
  tpu_gssh 120 "tail -c 20000 $GDP/$GD/out/export.log 2>/dev/null" > "$OUT/grade_export.log" || true
  if [ "$(tr -dc 0-9 < "$OUT/grade_export.rc" 2>/dev/null)" = "0" ] && tpu_gscp_from "$GDP/$GD/out/weights.npz" "$OUT/grade_weights.npz" 2400; then
    echo "[grade] weights npz secured ($(stat -c %s "$OUT/grade_weights.npz" 2>/dev/null) bytes)"
  else
    echo "[grade] npz export or pull failed (export rc=$(tr -dc 0-9 < "$OUT/grade_export.rc" 2>/dev/null); see grade_export.log); stage 2 gets no file"
  fi
}
cleanup_gd() { if [ "${XSTACK_KEEP_GRADE_DIR:-0}" != "1" ]; then tpu_gssh 120 "rm -rf $GDP/$GD" || true; else echo "[grade] XSTACK_KEEP_GRADE_DIR=1: grade dir kept at $GDP/$GD"; fi; }

RC=""; DEADPOLLS=0; EMPTYPOLLS=0; RUN_SECS=0; NOTPU_POLLS=0
while :; do
  ELAPSED=$(( $(date +%s) - START ))
  if [ "${CAP_S:-0}" -gt 0 ] 2>/dev/null && [ "$ELAPSED" -gt "$CAP_S" ]; then
    echo "[grade] CAP: graded run alive ${ELAPSED}s > cap ${CAP_S}s; killing and harvesting"
    kill_run; sleep 5
    fetch_outputs
    echo "graded run killed at ${ELAPSED}s (cap ${CAP_S}s = 10x the expected wall, at least 2 h)" > "$OUT/grade_capped"
    RC=127
    npz_export_pull   # a capped run's checkpoint is exported before the grade directory goes
    cleanup_gd
    emit_row 127 "CAPPED: graded run killed at ${ELAPSED}s (cap ${CAP_S}s); metrics harvested if the migration wrote any" "$MOUT" "$ELAPSED"
    exit 1
  fi
  if [ "$ELAPSED" -gt "$POLL_BUDGET_S" ]; then
    kill_run
    fetch_outputs
    cleanup_gd
    emit_row 124 "poll budget ${POLL_BUDGET_S}s exceeded; killed" "$MOUT" "$ELAPSED"
    exit 1
  fi
  RC="$(tpu_gssh 90 "if [ -f $GDP/$GD/out/rc ]; then r=\$(tr -dc 0-9 < $GDP/$GD/out/rc); echo \${r:-125}; fi")"
  RC="$(echo "$RC" | tr -dc 0-9)"
  if [ -n "$RC" ]; then break; fi
  ALIVE="$(tpu_gssh 90 "pgrep -f 'setup_en[v].sh|run_trai[n].py' >/dev/null && echo Y || echo N")"
  if [ "$ALIVE" = "N" ]; then
    DEADPOLLS=$(( DEADPOLLS + 1 )); EMPTYPOLLS=0
    if [ "$DEADPOLLS" -ge 2 ]; then RC=125; echo "[grade] no rc and no processes: crashed without an rc marker"; break; fi
  elif [ -z "$ALIVE" ]; then
    # ssh returned nothing: a flaky link or a dead machine; sustained silence is the machine gone
    EMPTYPOLLS=$(( EMPTYPOLLS + 1 ))
    if [ "$EMPTYPOLLS" -ge 15 ]; then RC=126; echo "[grade] machine unreachable for about 30 min (infra)"; break; fi
  else
    DEADPOLLS=0; EMPTYPOLLS=0
    # A graded run that never acquires the TPU is a CPU fallback and can crawl for days. Once
    # run_train.py has been alive for the grace period with no libtpu lock and no accelerator device
    # open, on two consecutive polls, it is killed and the turn fails (the next message says so).
    if tpu_gssh 60 "pgrep -f 'run_trai[n].py' >/dev/null" >/dev/null 2>&1; then
      RUN_SECS=$(( RUN_SECS + 120 ))
      if [ "$RUN_SECS" -ge "${XSTACK_NOTPU_GRACE_S:-7200}" ]; then
        u="$(tpu_gssh 90 "test -e /tmp/libtpu_lock && echo TPU; for p in \$(pgrep -u \$(id -un) -f 'pytho[n]'); do ls -l /proc/\$p/fd 2>/dev/null | grep -qE '/dev/(accel|vfio)' && echo TPU && break; done" | grep -c TPU)"
        if [ "${u:-0}" -eq 0 ]; then NOTPU_POLLS=$(( NOTPU_POLLS + 1 )); else NOTPU_POLLS=0; fi
        if [ "$NOTPU_POLLS" -ge 2 ]; then
          echo "[grade] NO_TPU: run alive ${RUN_SECS}s without acquiring the TPU (CPU fallback); killing, the turn fails"
          kill_run
          tpu_gssh 300 "cat $GDP/$GD/out/grade.log 2>/dev/null" > "$LOUT" || true
          : > "$MOUT"
          cleanup_gd
          echo "NO_TPU run alive ${RUN_SECS}s with no TPU acquisition; killed" > "$OUT/grade_notpu"
          emit_row 1 "NO_TPU: graded run never acquired the TPU (CPU fallback); killed after ${RUN_SECS}s" "$MOUT" "$ELAPSED"
          exit 1
        fi
      fi
    fi
  fi
  sleep 120
  echo "[grade] running... ${ELAPSED}s elapsed; log tail:"
  TAILTXT="$(tpu_gssh 90 "tail -c 300 $GDP/$GD/out/grade.log 2>/dev/null" | tail -3)"
  echo "$TAILTXT"
  [ -n "$TAILTXT" ] && printf '[%ss] %s\n' "$ELAPSED" "$TAILTXT" >> "$OUT/grade_progress.log"
done
WALL=$(( $(date +%s) - START ))

fetch_outputs
[ -s "$LOUT" ] || echo "[grade] WARNING grade.log unrecoverable"
if [ "$RC" = "0" ]; then npz_export_pull; fi
cleanup_gd

NOTE=""
[ "$RC" != "0" ] && NOTE="run exited $RC; see grade_full.log"
# A TPU runtime or machine failure in the log is not a verdict: the driver reruns the grade.
if [ "$RC" != "0" ] && grep -qE "Slice health check failed|Halt is unexpected|Worker connection .* failed|disconnected from the controller" "$LOUT" 2>/dev/null; then
  if ! grep -qE "Traceback|Error:.*(line|module)" "$LOUT" 2>/dev/null || grep -qE "Slice health check failed|Halt is unexpected" "$LOUT" 2>/dev/null; then
    RC=126; NOTE="infra: TPU runtime failure signature in the grade log (not a verdict)"
  fi
fi
emit_row "$RC" "$NOTE" "$MOUT" "$WALL"
case "$RC" in 0) exit 0;; 126) exit 126;; *) exit 1;; esac
