#!/usr/bin/env python3
"""Token accounting from a Claude Code stream-json trace (claude-trace.jsonl).

Drop-in interface twin of parse_rollout.py for cells whose agent is the
Claude Code launcher: same CLI, same summary JSON schema, so goal
drivers can account either agent. The trace may be several invocations
appended end-to-end (one per goal-loop turn); sums are cumulative across all
of them, matching codex's monotone-across-resume semantics.

Field mapping (Anthropic usage -> codex schema):
  fresh_input_tokens    = sum(input_tokens + cache_creation_input_tokens)
  cached_input_tokens   = sum(cache_read_input_tokens)
  input_tokens          = fresh + cached
  output_tokens         = sum(output_tokens)
  reasoning_output_tokens = 0 (thinking tokens are not split out by the API;
                             recorded as 0, see reasoning_note)
  peak_context_tokens   = max over requests of (input + cache_read +
                          cache_creation + output)
Compactions: system events whose subtype mentions "compact".

Usage: parse_claude_trace.py TRACE.jsonl [--merge-cell CELL_JSON]
                             [--timeline TIMELINE_JSON]
"""

import argparse
import json
import sys


def parse_trace(path):
    """Accounting source: RESULT events (one per shim invocation / goal
    turn) — z.ai zeroes the per-assistant-message usage but reports real
    cumulative usage per invocation on the result event, including a
    thinking_tokens split."""
    fresh = cached = out = reasoning = 0
    api_messages = 0
    peak = 0
    compactions = 0
    timeline = []
    models = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "system" and "compact" in str(ev.get("subtype", "")).lower():
                compactions += 1
            if ev.get("type") == "assistant":
                api_messages += 1
                model = (ev.get("message") or {}).get("model")
                if model:
                    models.add(model)
                continue
            if ev.get("type") != "result":
                continue
            usage = ev.get("usage")
            if not isinstance(usage, dict):
                continue
            inp = usage.get("input_tokens") or 0
            ccr = usage.get("cache_creation_input_tokens") or 0
            crd = usage.get("cache_read_input_tokens") or 0
            o = usage.get("output_tokens") or 0
            think = (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0
            fresh += inp + ccr
            cached += crd
            out += o
            reasoning += think
            ctx = inp + ccr + crd + o
            peak = max(peak, ctx)
            timeline.append(
                {
                    "ts": None,
                    "input": inp + ccr,
                    "cached_input": crd,
                    "output": o,
                    "reasoning_output": think,
                    "total": ctx,
                    "cost_usd": ev.get("total_cost_usd"),
                }
            )

    summary = {
        "rollout_events": len(timeline),
        "compaction_events": compactions,
        "input_tokens": fresh + cached,
        "cached_input_tokens": cached,
        "fresh_input_tokens": fresh,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
        "total_tokens": fresh + cached + out,
        "peak_context_tokens": peak,
        "requests": len(timeline),
        "api_messages": api_messages,
        "agent": "claude-harness",
    }
    if models:
        summary["models_seen"] = sorted(models)
    return summary, timeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollout")
    ap.add_argument("--merge-cell")
    ap.add_argument("--timeline")
    args = ap.parse_args()

    summary, timeline = parse_trace(args.rollout)
    print(json.dumps(summary, indent=1))

    if args.timeline:
        with open(args.timeline, "w", encoding="utf-8") as fh:
            json.dump(timeline, fh)
    if args.merge_cell:
        try:
            with open(args.merge_cell, encoding="utf-8") as fh:
                cell = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"parse_claude_trace: cannot merge into cell.json: {exc}", file=sys.stderr)
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
