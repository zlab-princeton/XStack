#!/usr/bin/env python3
"""render_prompts.py [--check] [REPO ...]

Render the two agent-facing texts of each task from its criteria file and the two templates:

  tasks/<repo>/prompt.md   prompts/phase1.template.md, {{REPO_CONTRACT}} <- criteria contract_lines
  tasks/<repo>/phase2.md   prompts/phase2.template.md, {{RECIPE}}        <- criteria phase2_recipe

The phase-2 text passes a leak gate before it is written: it must not contain the bar value,
any decimal number of the bar sentence, or grading vocabulary (graded, hidden, criteria, bar,
tolerance, anchor, published, threshold, benchmark, band).

--check renders in memory and compares with the files in the tree (exit 1 on any difference);
without it the files are written. Default: every task.
"""
import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
TPL1 = ROOT / "prompts" / "phase1.template.md"
TPL2 = ROOT / "prompts" / "phase2.template.md"
VOCAB = re.compile(
    r"\b(graded?|grader|grading|hidden|criteria|benchmark|band|tolerance|anchor|published|threshold)\b", re.I)
CEILINGS = {0.0, 1.0, 100.0, 1000.0}   # scale ceilings are not secrets (0-1 fraction, percent, DMControl return)


def numbers(text):
    out = []
    for tok in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])", text or ""):
        with contextlib.suppress(ValueError):
            out.append(float(tok.replace(",", "")))
    return out


def forbidden_numbers(crit):
    vals = set()
    for k in ("bar", "bar2"):
        v = crit.get(k)
        if v not in (None, ""):
            with contextlib.suppress(ValueError):
                vals.add(float(v))
    for tok in re.findall(r"\d[\d,]*\.\d+", crit.get("bar_sentence", "") or ""):
        with contextlib.suppress(ValueError):
            vals.add(float(tok.replace(",", "")))
    return {v for v in vals if v not in CEILINGS}


def leak_gate(text, crit):
    bad = []
    forb = forbidden_numbers(crit)
    for n in numbers(text):
        if any(abs(n - f) < 1e-9 for f in forb):
            bad.append(f"bar-linked number {n:g} present")
    for m in VOCAB.finditer(text):
        bad.append(f"grading vocabulary {m.group(0)!r} at offset {m.start()}")
    return bad


def render(repo):
    crit = json.loads((TASKS / repo / "criteria.json").read_text())
    lines = crit.get("contract_lines", [])
    block = ("\n".join(lines) + "\n\n") if lines else "\n"
    p1 = TPL1.read_text().replace("{{REPO_CONTRACT}}\n", block)
    p2 = TPL2.read_text().replace("{{RECIPE}}", (crit.get("phase2_recipe") or "").strip())
    for t, name in ((p1, "phase 1"), (p2, "phase 2")):
        if "{{" in t:
            raise SystemExit(f"{repo}: unfilled slot in the {name} template")
    bad = leak_gate(p2, crit)
    if bad:
        raise SystemExit(f"{repo}: phase-2 leak gate failed: " + "; ".join(bad))
    return p1, p2


def main(argv):
    check = "--check" in argv
    repos = [a for a in argv if not a.startswith("--")] or sorted(p.name for p in TASKS.iterdir() if (p / "criteria.json").exists())
    bad = 0
    for repo in repos:
        p1, p2 = render(repo)
        for text, name in ((p1, "prompt.md"), (p2, "phase2.md")):
            f = TASKS / repo / name
            if check:
                same = f.exists() and f.read_text() == text
                bad += 0 if same else 1
                print(f"{repo}/{name}: {'same' if same else 'DIFFERS'}")
            else:
                f.write_text(text)
                print(f"{repo}/{name}: written ({len(text)} chars)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
