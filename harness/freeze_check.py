#!/usr/bin/env python3
"""Phase-2 freeze classifier.

stdin: `git diff --name-status BASE HEAD` of the frozen turn.
argv[1]: the file(s) the agent may change (the entry, run_train.py; a comma list for a repo
         whose grade command has two entries, e.g. nanovllm run_bench.py,score_logprobs.py).
stdout, first line: VIOLATION | ENTRY | UNCHANGED; then the offending lines.

Ignored: the harness's own files (codex-*-turnN.txt, TPU_ACCESS.md, and the Claude Code shim's claude-trace.jsonl / shim-claude-stderr.txt /
.shim_idle_killed / .shim_parked / .shim_cap_waits) and
caches. Tolerated: ADDED notes/logs (*.md, *.log). Everything else that is
modified, deleted, renamed, or added is a violation.
"""
import re
import sys

entries = (sys.argv[1] if len(sys.argv) > 1 else "run_train.py").split(",")
harness = re.compile(r"^(codex-(stdout|stderr|final)-turn\d+\.txt|TPU_ACCESS\.md|claude-trace\.jsonl|shim-claude-stderr\.txt|\.shim_idle_killed|\.shim_parked|\.shim_cap_waits)$")  # the launchers' own files: the Claude Code transcript, stderr and markers live in the workspace
ignore = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache)(/|$)|\.pyc$")
notes = re.compile(r"\.(md|log)$", re.I)
viol, entry_changed = [], False
for line in sys.stdin.read().splitlines():
    parts = line.split("\t")
    if len(parts) < 2:
        continue
    st, paths = parts[0], parts[1:]
    if all(harness.match(p) or ignore.search(p) for p in paths):
        continue
    if len(paths) == 1 and paths[0] in entries and st[0] in "AM":
        entry_changed = True
        continue
    if st[0] == "A" and all(notes.search(p) for p in paths):
        continue
    viol.append(line.replace("\t", " "))
print("VIOLATION" if viol else ("ENTRY" if entry_changed else "UNCHANGED"))
for v in viol[:40]:
    print(v)
