#!/usr/bin/env bash
# Phase 1 of one cell: the blind turn, graded once. Sets the task's grading
# criteria in the environment and hands over to the driver (run_cell.sh).
# usage: phase1.sh REPO MODEL EFFORT
#   env: XSTACK_RUN   results/<run> (default run1)
#        XSTACK_AGENT_LAUNCHER  the agent launcher (default harness/agents/codex.sh)
#        XSTACK_AGENT_KIND      codex (default) or claude: how the agent's home is seeded and preserved
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/harness/tpu_env.sh"
REPO="${1:?usage: phase1.sh REPO MODEL EFFORT}"; MODEL="${2:?}"; EFFORT="${3:?}"
TASK_DIR="tasks/$REPO"
CRIT="$ROOT/$TASK_DIR/criteria.json"
[ -f "$CRIT" ] || { echo "phase1: no task $REPO ($CRIT)" >&2; exit 2; }
[ -d "$ROOT/$TASK_DIR/source_torch" ] || { echo "phase1: source not fetched: python3 fetch_sources.py $REPO" >&2; exit 2; }

get() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$CRIT" "$1"; }

export XSTACK_PROMPT="$TASK_DIR/prompt.md"
export XSTACK_FAIL_MSG="prompts/fail.md"
export XSTACK_FAIL_NOTPU_MSG="prompts/fail_notpu.md"
export XSTACK_GRADE_CMD="$(get grade_cmd)"
export XSTACK_TASK="$REPO"
export XSTACK_BAR="$(get bar)"
export XSTACK_BAR_CMP="$(get bar_cmp)"
B2="$(get bar2)"; [ -n "$B2" ] && export XSTACK_BAR2="$B2"
MR="$(get metric_re)"; [ -n "$MR" ] && export XSTACK_METRIC_RE="$MR"
DR="$(get dataset_regex)"; [ -n "$DR" ] && export XSTACK_DATASET_RE="$DR"
export XSTACK_CRITERION="phase 1 (repo $REPO): one blind turn, graded once on a fresh machine; PASS iff a metric matching /$(get metric_re)/ is $(get bar_cmp) $(get bar)${B2:+..$B2} ($(get bar_sentence), $(get protocol_sentence))"

exec "$ROOT/harness/run_cell.sh" "${XSTACK_RUN:-run1}" "$TASK_DIR" "$MODEL" "$EFFORT" 1
