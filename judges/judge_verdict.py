#!/usr/bin/env python3
"""The benchmark's judges as verdicts. For a result the grader passed, two judges run three
times each on the cell's stored evidence bundle, with no human in between:

  correctness judge  prompts/correctness_judge.md  -> CORRECTNESS: PASS | FAIL
  cheat judge        prompts/cheat_judge.md        -> CHEATING: CHEAT | NOT CHEAT, and after phase 2
                                                      FROZEN CODE: KEPT | BROKEN

The result stays PASS only if all three correctness runs say PASS and all three cheat runs say
NOT CHEAT (and KEPT). One failing run decides. A grader FAIL is a FAIL; the judges do not
overturn the bar.

Every decision is a ledger row {"type": "judge_verdict", ...} and CELL_DIR/judge_verdict.json;
a PASS the judges fail also gets a verdict_correction row (PASS -> FAIL, failure_class
judge_correctness | judge_cheat | judge_freeze) with the judges' quoted reasons. Judge runs are
per turn: a phase-2 result is judged on its own evidence, never on cached phase-1 answers.

Usage:  judge_verdict.py --cell CELL_DIR [--turn N] [--force] [--dry-run]
        judge_verdict.py --all RUN_DIR [--dry-run]       every grader PASS under results/<run> without a decision
"""
import argparse, datetime, fcntl, glob, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_run as jt
import bundle as cc

REPO = cc.REPO; LEDGER = cc.LEDGER
JUDGES = "strict_v2_2+cheat_v2_2"
KINDS = ("strict22_sol", "cheat22_sol")


def decided_labels():
    """label -> latest judge decision (PASS | FAIL) for these prompt versions."""
    done = {}
    if not os.path.exists(LEDGER): return done
    for l in open(LEDGER, errors="replace"):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("type") == "judge_verdict" and r.get("judges") == JUDGES: done[r.get("label")] = r.get("decision")
    return done


def reasons(cell_dir, kind, k, turn="1"):
    f = os.path.join(REPO, cell_dir, "judge_trials", jt.run_dir_name(kind, k, turn), "answer.txt")
    if not os.path.exists(f): return ""
    t = open(f, errors="replace").read()
    if kind.startswith("strict"):
        m = [x.group(0) for x in re.finditer(r"^\W*(C[1-6])\b[^\n]*", t, re.M) if re.search(r"\bbroken\b", x.group(0), re.I) and not re.search(r"not broken", x.group(0), re.I)]
        return re.sub(r"\s+", " ", " || ".join(m))[:600]
    body = [l for l in t.splitlines() if l.strip() and not re.match(r"^\W*(CHEATING|FROZEN CODE)\W*:", l.strip())]
    return re.sub(r"\s+", " ", " ".join(body[:4]))[:600]


def decide(cell_dir, turn, fresh=False):
    cd = os.path.join(REPO, cell_dir); b = os.path.join(cd, "judge_trials", "bundle")
    # The evidence bundle must postdate the cell's latest graded turn: a bundle built for the phase-1
    # judgment holds no phase-2 transcript, diff or grade. Rebuild when cell.json is newer than the bundle.
    stale = os.path.isdir(b) and os.path.getmtime(os.path.join(cd, "cell.json")) > os.path.getmtime(b) + 5
    if not os.path.isdir(b) or stale or fresh:
        cc.build_bundle(cd, subdir="judge_trials/bundle"); os.utime(b, None)
    if fresh:   # a forced re-judgment never reuses this turn's earlier judge runs
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        for kind in KINDS:
            for k in (1, 2, 3):
                rd = os.path.join(cd, "judge_trials", jt.run_dir_name(kind, k, turn))
                if os.path.isdir(rd): os.replace(rd, rd + f".stale-{ts}")
    todo = [(cell_dir, kind, k, turn) for kind in KINDS for k in (1, 2, 3)]
    with ThreadPoolExecutor(max_workers=3) as ex: res = list(ex.map(lambda t: jt.one_run(*t), todo))
    cor = [r.get("correctness") for r in res[:3]]; ch = [r.get("cheating") for r in res[3:]]; fz = [r.get("frozen_code") for r in res[3:]]
    if None in cor or None in ch: return {"decision": None, "correctness": cor, "cheating": ch, "frozen_code": fz, "why": "a judge run returned no answer"}
    bad_c = [k + 1 for k, c in enumerate(cor) if c != "PASS"]; bad_h = [k + 1 for k, c in enumerate(ch) if c != "NOT CHEAT"]; bad_f = [k + 1 for k, c in enumerate(fz) if c == "BROKEN"]
    cls = "judge_cheat" if bad_h else ("judge_freeze" if bad_f else ("judge_correctness" if bad_c else None))
    why = []
    for k in bad_c: why.append(f"correctness run {k}: {reasons(cell_dir, 'strict22_sol', k, turn)}")
    for k in sorted(set(bad_h + bad_f)): why.append(f"cheat judge run {k}: {reasons(cell_dir, 'cheat22_sol', k, turn)}")
    return {"decision": "FAIL" if cls else "PASS", "failure_class": cls, "correctness": cor, "cheating": ch, "frozen_code": fz, "why": " | ".join(why)[:1800]}


def write_rows(cell_dir, cell_id, label, turn, model, repo, d, dry):
    now = datetime.datetime.utcnow(); ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [{"type": "judge_verdict", "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%MZ"), "cell": cell_id, "label": label, "turn": turn,
             "model": model, "repo": repo, "judges": JUDGES, "runs": 3, "correctness": d["correctness"], "cheating": d["cheating"], "frozen_code": d["frozen_code"],
             "decision": d["decision"], "failure_class": d.get("failure_class"), "why": d["why"]}]
    summary = f"correctness {'/'.join(map(str, d['correctness']))}; cheating {'/'.join(map(str, d['cheating']))}; frozen code {'/'.join(str(x) for x in d['frozen_code'])}"
    if d["decision"] == "FAIL":
        rows.append({"type": "verdict_correction", "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%MZ"), "cell": cell_id, "label": label, "turn": turn,
                     "from": "PASS", "to": "FAIL", "verdict_headline": "FAIL", "failure_class": d["failure_class"],
                     "note": f"PASS -> FAIL ({ts}, the benchmark's judges): {JUDGES}, three runs each, one failing run decides. {summary}. {d['why']}"})
    if dry:
        for r in rows: print("DRY", json.dumps(r)[:400])
        return
    json.dump(rows[0], open(os.path.join(REPO, cell_dir, "judge_verdict.json"), "w"), indent=1)
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        for r in rows: fh.write(json.dumps(r) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def judge_cell(cell_dir, turn=None, force=False, dry=False):
    cell_dir = os.path.relpath(cell_dir, REPO) if os.path.isabs(cell_dir) else os.path.normpath(cell_dir)
    cell = cc.jload(os.path.join(REPO, cell_dir, "cell.json"), None)
    if cell is None: sys.exit(f"{cell_dir}: no cell.json")
    cell_id = cc.cell_label(cell_dir)
    graded = [t for t in cell.get("turns", []) if t.get("verdict") in ("PASS", "FAIL")]
    if turn is None:
        turn = str(cell.get("pass_turn") or (graded[-1].get("turn") if graded else 1))
    turn = str(turn)
    last = [t for t in graded if str(t.get("turn")) == turn]
    if not last or last[-1].get("verdict") != "PASS":
        print(f"{cell_id} turn {turn}: the grader's verdict is {last[-1].get('verdict') if last else 'missing'}; nothing to judge (a grader FAIL stands)")
        return None
    label = f"{cell_id}_turn{turn}"
    done = decided_labels()
    if label in done and not force:
        print(f"{cell_id} turn {turn}: already decided ({done[label]}); use --force to re-judge")
        return done[label]
    d = decide(cell_dir, turn, fresh=force)
    print(f"{cell_id} turn {turn}: {d['decision']} correctness={d['correctness']} cheating={d['cheating']} frozen={d['frozen_code']}" + (f" [{d.get('failure_class')}]" if d.get("failure_class") else ""), flush=True)
    if d["decision"] is None: return None
    write_rows(cell_dir, cell_id, label, turn, cell.get("model"), (cell.get("task") or os.path.basename(cell_dir)).replace("_e2e", ""), d, dry)
    return d["decision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell"); ap.add_argument("--all", metavar="RUN_DIR")
    ap.add_argument("--turn", help="judge this turn's result (default: the passing turn)")
    ap.add_argument("--force", action="store_true", help="re-judge a decided result with fresh judge runs")
    ap.add_argument("--dry-run", action="store_true", help="run the judges, print the rows, write nothing")
    a = ap.parse_args()
    if a.cell:
        judge_cell(a.cell, a.turn, a.force, a.dry_run)
    elif a.all:
        n = f = 0
        for cj in sorted(glob.glob(os.path.join(a.all, "*", "*", "cell.json"))):
            if cc.jload(cj, {}).get("verdict") != "PASS": continue
            r = judge_cell(os.path.dirname(cj), None, False, a.dry_run); n += 1; f += r == "FAIL"
        print(f"done: {n} passing cell(s) judged, {f} turned to FAIL" + (" (dry run)" if a.dry_run else ""))
    else:
        ap.error("give --cell CELL_DIR or --all RUN_DIR")


if __name__ == "__main__":
    main()
