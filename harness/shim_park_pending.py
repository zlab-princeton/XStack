#!/usr/bin/env python3
"""Pending ScheduleWakeup park in a Claude Code stream-json trace.

Prints the number of seconds for which a silent stream is still a waiting agent rather than
a hung one, or 0. Two shapes, both requiring that the stream's last `result` event is a
success and nothing from the model or the CLI follows it (no assistant/user/system event: a
fired wakeup or task notification starts a new segment with a `system` init):

* park: the segment that produced that result holds a ScheduleWakeup tool
  result ("Next wakeup scheduled for HH:MM:SS (in Ns) ..."): prints N (the CLI clamps it to
  3600 s);
* linger: no such tool result, the agent signed off while an agent-launched
  background task keeps the CLI alive (it re-invokes the model on each task event): prints
  the linger allowance (7200 s).

Callers add their own margin and kill once the allowance is overdue with no new event.
Reads only the tail of the file (the cumulative trace can be hundreds of MB).

Usage: shim_park_pending.py <trace.jsonl> [tail_bytes]
Caller: the idle watchdog in harness/agents/claude.sh.
"""
import json
import os
import re
import sys

WAKEUP_RE = re.compile(r"Next wakeup scheduled for .*?\(in (\d+)s\)")


def park_pending(path, tail_bytes=4_000_000):
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    rows = []
    with open(path, "rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
            f.readline()  # drop the partial line
        for line in f:
            try:
                rows.append(json.loads(line.decode("utf-8", "replace")))
            except Exception:
                pass
    res = [i for i, e in enumerate(rows) if e.get("type") == "result"]
    if not res:
        return 0
    r = res[-1]
    if rows[r].get("is_error"):
        return 0
    if any(e.get("type") in ("assistant", "user", "system") for e in rows[r + 1:]):
        return 0
    prev = res[-2] if len(res) > 1 else -1
    n = 0
    for e in rows[prev + 1:r]:
        m = e.get("message") or {}
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                mm = WAKEUP_RE.search(str(b.get("content")))
                if mm:
                    n = int(mm.group(1))
    if n:
        return min(n, 3600)
    return linger_allow()


def linger_allow():
    return 7200


if __name__ == "__main__":
    tb = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
    print(park_pending(sys.argv[1], tb))
