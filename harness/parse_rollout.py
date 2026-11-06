#!/usr/bin/env python3
"""Extract exact per-session token accounting from a codex rollout JSONL.

The rollout (persisted when the cell runs with a per-cell CODEX_HOME and
without --ephemeral) carries per-request
token_count events with the full split the exec footer collapses:
fresh vs cached input, output, reasoning, and the live context size of
every request.

Usage:
    parse_rollout.py ROLLOUT.jsonl [--merge-cell CELL_JSON]
                           [--timeline TIMELINE_JSON]

Prints the summary JSON to stdout. --merge-cell inserts it under the
"tokens_exact" key of an existing cell.json (plus top-level convenience
fields). --timeline writes the per-request series for the explorer.
Never raises on schema drift: unknown shapes degrade to partial output.
"""

import argparse
import json
import sys


def _get(d, *path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def parse_rollout(path):
    totals = None
    context_window = None
    timeline = []
    compactions = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = str(entry.get("type", ""))
            payload = entry.get("payload") or {}
            ptype = str(payload.get("type", ""))
            if "compact" in etype.lower() or "compact" in ptype.lower():
                compactions += 1
            info = None
            if ptype == "token_count":
                info = payload.get("info") or {}
            elif etype == "token_count":
                info = entry.get("info") or payload or {}
            if not info:
                continue
            total = info.get("total_token_usage") or {}
            last = info.get("last_token_usage") or {}
            context_window = info.get("model_context_window") or context_window
            if total:
                totals = total
            if last:
                timeline.append(
                    {
                        "ts": entry.get("timestamp"),
                        "input": last.get("input_tokens"),
                        "cached_input": last.get("cached_input_tokens"),
                        "output": last.get("output_tokens"),
                        "reasoning_output": last.get("reasoning_output_tokens"),
                        "total": last.get("total_tokens"),
                    }
                )

    summary = {"rollout_events": len(timeline), "compaction_events": compactions}
    if context_window:
        summary["model_context_window"] = context_window
    if totals:
        inp = totals.get("input_tokens")
        cached = totals.get("cached_input_tokens")
        summary.update(
            {
                "input_tokens": inp,
                "cached_input_tokens": cached,
                "fresh_input_tokens": (inp - cached)
                if isinstance(inp, int) and isinstance(cached, int)
                else None,
                "output_tokens": totals.get("output_tokens"),
                "reasoning_output_tokens": totals.get("reasoning_output_tokens"),
                "total_tokens": totals.get("total_tokens"),
            }
        )
    requests = [t for t in timeline if isinstance(t.get("input"), int)]
    if requests:
        peak = max(requests, key=lambda t: t["input"])
        summary["peak_context_tokens"] = peak["input"] + (peak.get("output") or 0)
        summary["requests"] = len(requests)
    return summary, timeline


def _is_claude_trace(path):
    """Claude Code stream-json (the Claude Code launcher's rollout equivalent) starts with
    a system/init event carrying session_id; codex rollouts never do."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return ev.get("type") == "system" and "session_id" in ev
    except OSError:
        return False
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollout")
    ap.add_argument("--merge-cell")
    ap.add_argument("--timeline")
    args = ap.parse_args()

    if _is_claude_trace(args.rollout):
        # delegate: same CLI, same output schema (agent-neutral drivers)
        import os
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "parse_claude_trace.py")
        os.execv(sys.executable, [sys.executable, script] + sys.argv[1:])

    summary, timeline = parse_rollout(args.rollout)
    print(json.dumps(summary, indent=1))

    if args.timeline:
        with open(args.timeline, "w", encoding="utf-8") as fh:
            json.dump(timeline, fh)
    if args.merge_cell:
        try:
            with open(args.merge_cell, encoding="utf-8") as fh:
                cell = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"parse_codex_rollout: cannot merge into cell.json: {exc}", file=sys.stderr)
            return
        cell["tokens_exact"] = summary
        for key in ("output_tokens", "reasoning_output_tokens", "fresh_input_tokens",
                    "cached_input_tokens", "peak_context_tokens"):
            if summary.get(key) is not None:
                cell[key] = summary[key]
        with open(args.merge_cell, "w", encoding="utf-8") as fh:
            json.dump(cell, fh, indent=1)


if __name__ == "__main__":
    main()
