#!/usr/bin/env bash
# Phase 2 of one cell: after a phase-1 FAIL, the same agent session gets one
# more turn with the task's phase-2 message (the recipe of the main experiment,
# never the bar) while the tree is frozen: only run_train.py may change. The
# driver's freeze check fails any other change (freeze_violation, ungraded);
# otherwise the same grader and criteria as phase 1 apply, on a fresh machine.
# The cell's verdict becomes the phase-2 verdict.
# usage: phase2.sh REPO MODEL EFFORT     (env as for phase1.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/harness/tpu_env.sh"
REPO="${1:?usage: phase2.sh REPO MODEL EFFORT}"; MODEL="${2:?}"; EFFORT="${3:?}"
TASK_DIR="tasks/$REPO"
CRIT="$ROOT/$TASK_DIR/criteria.json"
[ -f "$CRIT" ] || { echo "phase2: no task $REPO ($CRIT)" >&2; exit 2; }
P2_FILE="$TASK_DIR/phase2.md"
[ -f "$ROOT/$P2_FILE" ] || { echo "phase2: no phase-2 message $P2_FILE" >&2; exit 2; }

get() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$CRIT" "$1"; }

RUN_NAME="${XSTACK_RUN:-run1}"
OUTDIR="$ROOT/results/$RUN_NAME/${MODEL}_${EFFORT}/$REPO"
[ -d "$OUTDIR/codex_exec-0" ] || { echo "phase2: no phase-1 cell at $OUTDIR" >&2; exit 2; }
[ -s "$OUTDIR/session_id" ] || { echo "phase2: the phase-1 cell has no session_id (cannot resume)" >&2; exit 2; }

export XSTACK_PROMPT="$TASK_DIR/prompt.md"
# the turn-2 message IS the phase-2 message (both variants: a NO_TPU phase-1 detail must not swap in the blind fail text)
export XSTACK_FAIL_MSG="$P2_FILE"
export XSTACK_FAIL_NOTPU_MSG="$P2_FILE"
export XSTACK_GRADE_CMD="$(get grade_cmd)"
export XSTACK_TASK="$REPO"
export XSTACK_BAR="$(get bar)"
export XSTACK_BAR_CMP="$(get bar_cmp)"
B2="$(get bar2)"; [ -n "$B2" ] && export XSTACK_BAR2="$B2"
MR="$(get metric_re)"; [ -n "$MR" ] && export XSTACK_METRIC_RE="$MR"
DR="$(get dataset_regex)"; [ -n "$DR" ] && export XSTACK_DATASET_RE="$DR"
export XSTACK_CRITERION="phase 2 (repo $REPO): the phase-1 tree is frozen; turn 2 resumes the same session with the phase-2 message ($P2_FILE); any change outside the entry file = FAIL freeze_violation, ungraded; otherwise graded as phase 1: PASS iff a metric matching /$(get metric_re)/ is $(get bar_cmp) $(get bar)${B2:+..$B2} ($(get bar_sentence), $(get protocol_sentence))"
# freeze mode: the driver refuses any change outside the entry file at turn 2
export XSTACK_FREEZE=1
P2E="$(get phase2_entry)"; export XSTACK_FREEZE_ENTRY="${P2E:-run_train.py}"   # criteria may name the entry file(s)
# resume at turn 2 unless a grade-only resume is already set
export XSTACK_RESUME_FROM="${XSTACK_RESUME_FROM:-2}"

# Pre-turn hygiene: the resume path skips the fresh-cell reset, so give phase 2 a clean machine
# (skipped on a grade-only resume: the generation is already recorded).
if [ "${XSTACK_RESUME_GRADE_ONLY:-0}" != "1" ]; then
  bash "$ROOT/harness/tpu_reset.sh" || { echo "phase2: pre-turn reset failed (infra)" >&2; exit 3; }
fi

exec "$ROOT/harness/run_cell.sh" "$RUN_NAME" "$TASK_DIR" "$MODEL" "$EFFORT" 2
