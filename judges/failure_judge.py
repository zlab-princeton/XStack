#!/usr/bin/env python3
"""The failure judge: for a FAILED cell, classify why in the fixed taxonomy
(prompts/failure_taxonomy.md), cite the evidence, and say whether a
phase-2 (entry-only) change could plausibly have fixed it. Writes the ledger's
existing `turn_analysis` row shape, plus a human-readable report. Never changes a verdict.

Deliverable per cell:
  CELL_DIR/failure_analysis/       the evidence bundle the judge saw
  CELL_DIR/failure_analysis.json   the judge's answer
  CELL_DIR/failure_analysis.md     decision (class), reasoning, evidence, phase-2 call
  one ledger row  {"type": "turn_analysis", "failure_class": ..., "source": "failure_analysis_v1", ...}

Usage:
  failure_judge.py CELL_DIR [--turn N] [--bundle-only] [--force] [--model M] [--effort E]
  failure_judge.py --all RUN_DIR       every FAIL cell with no turn_analysis row yet
"""
import argparse, datetime, fcntl, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bundle as cc  # bundle builder, judge call, label helper

REPO = cc.REPO
LEDGER = cc.LEDGER
PROMPT = os.path.join(REPO, "prompts/failure_judge.md")
TAXONOMY = os.path.join(REPO, "prompts/failure_taxonomy.md")

recipe_text = cc.recipe_text   # shared with the judges' bundle

def has_row(label):
    if not os.path.exists(LEDGER):
        return False
    for l in open(LEDGER, errors="replace"):
        if '"turn_analysis"' in l and label in l: return True
    return False

def analyse(cell_dir, args):
    cell = cc.jload(os.path.join(cell_dir, "cell.json"), {})
    label = cc.cell_label(cell_dir)
    graded = [t for t in cell.get("turns", []) if t.get("verdict") in ("PASS", "FAIL")]
    # --turn N: analyse one turn's own FAIL, not the cell's latest graded turn (a cell whose phase 2 also
    # graded, or passed, still needs its turn-1 class for the failure-class figure); own bundle and report files.
    if args.turn:
        sel = [t for t in graded if int(t.get("turn") or 1) == args.turn]
        if not sel: print(f"skip {label}: no graded turn {args.turn}"); return
        last = sel[-1]
        if last.get("verdict") != "FAIL" and not args.force:
            print(f"skip {label}: turn {args.turn} verdict {last.get('verdict')} (use --force)"); return
        sub, out_json, out_md = f"failure_analysis_turn{args.turn}", f"failure_analysis_turn{args.turn}.json", f"failure_analysis_turn{args.turn}.md"
    else:
        last = graded[-1] if graded else {}
        sub, out_json, out_md = "failure_analysis", "failure_analysis.json", "failure_analysis.md"
        if cell.get("verdict") != "FAIL" and not args.force:
            print(f"skip {label}: verdict {cell.get('verdict')} (use --force)"); return
    if has_row(label) and not args.force:
        print(f"skip {label}: turn_analysis row exists (use --force)"); return
    b = cc.build_bundle(cell_dir, subdir=sub)
    if args.turn:
        with open(os.path.join(b, "context.txt"), "a") as fh:
            fh.write(f"result under analysis: the phase {args.turn} result (the number graded after phase {args.turn}); its driver verdict: {last.get('verdict')} ({last.get('verdict_detail')})."
                     + (" A phase 2 also ran afterwards; it is not the result under analysis. `run_train.py` here is the final file; `diff_phase2.patch` shows what phase 2 changed in it.\n"
                        if args.turn == 1 and os.path.exists(os.path.join(b, "diff_phase2.patch")) else "\n"))
    repo = (cell.get("task") or os.path.basename(cell_dir)).replace("_e2e", "")
    open(os.path.join(b, "recipe.txt"), "w").write(recipe_text(repo))
    tax = open(TAXONOMY, errors="replace").read() if os.path.exists(TAXONOMY) else (
        "pass, scope_miss (ran the wrong pipeline/variant/dataset or reported the wrong field), incomplete_transfer (a needed piece was not ported), "
        "fidelity_gap (right pipeline, number short of the bar), scale_deviation (wrong training length/batch/size), runtime_crash, no_tpu, "
        "wrong_fix (later turn changed the wrong thing), unchanged_tree, grader_artifact, fabricated_result, no_termination, infra")
    open(os.path.join(b, "taxonomy.md"), "w").write(tax)
    size = sum(os.path.getsize(os.path.join(b, f)) for f in os.listdir(b)) // 1024
    print(f"bundle: {b} ({size} KB)")
    if args.bundle_only: return
    ans, rc = cc.run_judge(b, PROMPT, args.model, args.effort)
    rec = {"model": args.model, "effort": args.effort, "date": datetime.datetime.utcnow().isoformat() + "Z", "rc": rc, "answer": ans, "turn": int(last.get("turn") or 1)}
    json.dump(rec, open(os.path.join(cell_dir, out_json), "w"), indent=1)
    md = [f"# Failure analysis: {label}" + (f" (turn {args.turn})" if args.turn else ""), "",
          f"**Class: {ans.get('failure_class', 'parse_error')}**" + (f" (also {ans['also']})" if ans.get("also") else "") + f" — phase-2 eligible: **{ans.get('phase2_eligible', '?')}**", "",
          f"Judge {args.model} at {args.effort} effort, {rec['date']}. A record of why, not a review of the verdict.", "",
          "## Root cause", "", str(ans.get("root_cause", "")), "", "## Evidence", ""] + [f"- {e}" for e in (ans.get("evidence") or ["(none quoted)"])] + [
          "", "## Metric vs bar", "", f"- metric: {ans.get('metric')}", f"- bar: {ans.get('bar')}", f"- driver reason: {last.get('verdict_detail')}", "",
          "## Phase-2 eligibility", "", f"{ans.get('phase2_eligible')}: {ans.get('phase2_reason', '')}", "",
          "## What the port got right", "", str(ans.get("port_got_right", ""))]
    if "parse_error" in ans: md += ["", "## Judge output could not be parsed", "", "```", str(ans.get("parse_error"))[:1500], "```"]
    open(os.path.join(cell_dir, out_md), "w").write("\n".join(md) + "\n")
    row = {"type": "turn_analysis", "date": datetime.date.today().isoformat(),
           "model": f"{cell.get('model')}/{cell.get('effort')}", "repo": repo, "cell": label, "turn": int(last.get("turn") or 1),
           "verdict": last.get("verdict") if args.turn else cell.get("verdict"), "failure_class": ans.get("failure_class", "unclassified"), "also": ans.get("also"),
           "metric": ans.get("metric"), "evidence": str(ans.get("root_cause", ""))[:600],
           "phase2_eligible": ans.get("phase2_eligible"), "source": f"failure_analysis_v1 ({args.model}/{args.effort})",
           "report": os.path.relpath(os.path.join(cell_dir, out_md), REPO)}
    with open(LEDGER, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX); fh.write(json.dumps(row) + "\n"); fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"DONE {label}: class={row['failure_class']} also={row['also']} phase2={row['phase2_eligible']} metric={row['metric']} | {str(ans.get('root_cause',''))[:110]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir", nargs="?")
    ap.add_argument("--all", metavar="RUN_DIR")
    ap.add_argument("--bundle-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--turn", type=int, help="analyse this turn's own FAIL (own bundle and report files: failure_analysis_turnN.*)")
    ap.add_argument("--model", default=os.environ.get("XSTACK_JUDGE_MODEL", "gpt-5.6-sol"))
    ap.add_argument("--effort", default=os.environ.get("XSTACK_JUDGE_EFFORT", "medium"))
    a = ap.parse_args()
    if a.all:
        for cj in sorted(glob.glob(os.path.join(a.all, "*", "*", "cell.json"))):
            analyse(os.path.dirname(cj), a)
    elif a.cell_dir: analyse(a.cell_dir, a)
    else: ap.error("CELL_DIR or --all RUN_DIR")

if __name__ == "__main__":
    main()
