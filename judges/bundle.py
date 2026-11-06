#!/usr/bin/env python3
"""The evidence bundle of one cell: the transcripts condensed to messages, commands and
output heads, the diffs, the graded entry file, the freeze check, the grade summary, the
metric lines of the graded log, the recipe, the source files the recipe names, and every
URL fetched and file written outside the migrated tree. The three judges read it:
judge_run.py (the correctness and cheat judges, driven by judge_verdict.py) and
failure_judge.py.

The work directory is XSTACK_WORK from the environment or from config.env at the repository root
(default <repo>/work); the ledger is $XSTACK_WORK/ledger.jsonl. The judges run through `codex`.

Usage:  bundle.py CELL_DIR [--subdir NAME]     write CELL_DIR/<NAME>/ (default judge_trials/bundle)
"""
import argparse, glob, json, os, re, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config_env():
    """config.env: KEY=VALUE lines; the environment wins, empty values fall through to the defaults."""
    path = os.path.join(REPO, "config.env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip(); value = value.strip().strip("'\"")
        if value and key not in os.environ:
            os.environ[key] = os.path.expanduser(value)


_load_config_env()
WORK = os.environ.get("XSTACK_WORK") or os.path.join(REPO, "work")
LEDGER = os.path.join(WORK, "ledger.jsonl")
CODEX_BIN = "codex"
JUDGE_HOME = os.path.join(WORK, "codex_homes", "judge")
CAP_OUT = 1200
CAP_PATCH = 120_000
CAP_TRANSCRIPT = 350_000
URL_RE = re.compile(r"https?://[^\s'\"<>)\\]+")
FETCH_RE = re.compile(r"\b(wget|curl|gdown|huggingface-cli|hf_hub_download|from_pretrained|torch\.hub|gsutil cp|gcloud storage cp|tar -x|unzip|pip download)\b")

def jload(p, d=None):
    try:
        with open(p) as fh: return json.load(fh)
    except Exception: return d

def cap(s, n):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s)-n} more chars]"

def cap_transcript(s):
    if len(s) <= CAP_TRANSCRIPT: return s
    head, tail = int(CAP_TRANSCRIPT * 0.6), int(CAP_TRANSCRIPT * 0.4)
    return s[:head] + f"\n\n... [transcript trimmed: {len(s)-head-tail} chars omitted from the middle] ...\n\n" + s[-tail:]

def compact(o, depth=0):
    if isinstance(o, dict): return {k: compact(v, depth + 1) for k, v in o.items()}
    if isinstance(o, list): return [compact(v, depth + 1) for v in o[:8]] + ([f"[... {len(o)-8} more]"] if len(o) > 8 else [])
    if isinstance(o, str): return cap(o, 600)
    return o

# ---- trajectory condensers ---------------------------------------------------
def condense_codex(rollout, turn_filter):
    """rollout.jsonl -> markdown of messages, commands and output heads.
    turn_filter: 1 = up to the first task_complete; 2 = everything after it."""
    out, turn = [], 1
    for line in open(rollout, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        p = e.get("payload") or {}
        t, pt = e.get("type"), p.get("type")
        if t == "event_msg" and pt == "task_complete":
            turn += 1; continue
        if (turn_filter == 1 and turn != 1) or (turn_filter == 2 and turn < 2): continue
        if t == "event_msg" and pt == "task_started" and turn > 1: out.append(f"\n---- turn {turn} begins ----\n")
        if t == "response_item" and pt == "message":
            role = p.get("role", "?")
            if role in ("developer", "system"): continue
            txt = " ".join(c.get("text", "") for c in p.get("content", []) if isinstance(c, dict))
            if role == "user" and txt.lstrip().startswith(("<recommended_plugins>", "<environment_context>", "<skills_instructions>", "<permissions")): continue
            if txt.strip(): out.append(f"**{role}:** {cap(txt, 3000 if role == 'user' else 2000)}")
        elif t == "response_item" and pt in ("custom_tool_call", "function_call"):
            cmd = p.get("input") or p.get("arguments") or ""
            try:
                a = json.loads(cmd) if isinstance(cmd, str) and cmd.startswith("{") else None
                if isinstance(a, dict): cmd = a.get("cmd") or a.get("command") or cmd
            except Exception: pass
            out.append(f"```\n$ {cap(cmd, 1500)}\n```")
        elif t == "response_item" and pt in ("custom_tool_call_output", "function_call_output"):
            o = p.get("output", "")
            if isinstance(o, str) and o.startswith("{"):
                try: o = json.loads(o).get("output", o)
                except Exception: pass
            out.append(f"<output>\n{cap(o, CAP_OUT)}\n</output>")
    return cap_transcript("\n\n".join(out))

def condense_claude(trace, turn_filter):
    if turn_filter != 1: return ""
    out = []
    for line in open(trace, errors="replace"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content", []) or []:
                if not isinstance(c, dict): continue
                if c.get("type") == "text" and c.get("text", "").strip(): out.append(f"**assistant:** {cap(c['text'], 2000)}")
                if c.get("type") == "tool_use":
                    inp = c.get("input") or {}
                    out.append(f"```\n$ {cap(inp.get('command') or json.dumps(inp), 1500)}\n```")
        elif e.get("type") == "user":
            for c in (e.get("message") or {}).get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    out.append(f"<output>\n{cap(c.get('content'), CAP_OUT)}\n</output>")
    return cap_transcript("\n\n".join(out))

def recipe_text(repo):
    """What the frozen turn is told (the rendered phase-2 message) or, failing that, the criteria fields."""
    p2 = os.path.join(REPO, f"tasks/{repo}/phase2.md")
    if os.path.exists(p2): return open(p2, errors="replace").read()[:6000]
    c = jload(os.path.join(REPO, f"tasks/{repo}/criteria.json"), {})
    return json.dumps({k: c.get(k) for k in ("bar", "bar_cmp", "metric_re", "dataset_regex", "dataset", "recipe", "contract", "bar_provenance", "expected_grade_wall_h") if c.get(k) is not None}, indent=1, default=str)[:6000]

def git(ws, *args):
    try: return subprocess.run(["git", "-C", ws, *args], capture_output=True, text=True, timeout=60).stdout
    except Exception as exc: return f"(git failed: {exc})"

SRC_EXT = (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".json")
SRC_CAP = 300_000          # bytes; larger files (annotation JSONs, transforms) are data, not code

def source_files_for(repo, recipe, out_dir):
    """Copy into out_dir every file the recipe text names (backticked paths, split on whitespace) that exists in the
    vendored source tree tasks/<repo>/source_torch/<one subdir>; returns the relative names copied."""
    roots = [d for d in glob.glob(os.path.join(REPO, f"tasks/{repo}/source_torch/*")) if os.path.isdir(d)]
    if not roots: return []
    root = roots[0]
    names = []
    for m in re.finditer(r"`([^`\n]+)`", recipe):
        for tok in m.group(1).split():
            tok = tok.strip("'\",;:()")
            if tok.endswith(SRC_EXT) and tok not in names: names.append(tok)
    found = []
    for tok in names:
        cand = os.path.join(root, tok)
        if not os.path.isfile(cand):
            hits = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f == os.path.basename(tok) and ".git" not in dp]
            hits = [h for h in hits if h.endswith(tok)] or hits
            cand = sorted(hits, key=len)[0] if hits else None
        if not cand or not os.path.isfile(cand) or os.path.getsize(cand) > SRC_CAP: continue
        rel = os.path.relpath(cand, root)
        if rel in found: continue
        dst = os.path.join(out_dir, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w").write(open(cand, errors="replace").read()); found.append(rel)
    if found:
        pin = ""
        for pf in ("SOURCE_COMMIT", "SOURCE.md", "source_commit.txt"):
            p = os.path.join(REPO, f"tasks/{repo}", pf)
            if os.path.exists(p): pin = open(p, errors="replace").read().strip()[:200]; break
        open(os.path.join(out_dir, "INDEX.txt"), "w").write(f"source: tasks/{repo}/source_torch/{os.path.basename(root)} (the fixed-commit PyTorch repository the model was given){(' — ' + pin) if pin else ''}\n" + "\n".join(found) + "\n")
    return found

def cell_label(cell_dir):
    parts = os.path.normpath(cell_dir).split(os.sep)
    return f"{parts[-3]}_{parts[-2].replace('_', '-')}_{parts[-1]}"

def build_bundle(cell_dir, subdir="judge_trials/bundle"):
    cell = jload(os.path.join(cell_dir, "cell.json"), {})
    ws = os.path.join(cell_dir, "codex_exec-0")
    b = os.path.join(cell_dir, subdir); os.makedirs(b, exist_ok=True)
    # Pick the trajectory by content, not by name: a Claude Code cell carries an empty
    # rollout.jsonl next to its real claude-trace.jsonl.
    traj, kind = None, None
    for p in (os.path.join(cell_dir, "rollout.jsonl"), os.path.join(ws, "claude-trace.jsonl"), os.path.join(cell_dir, "claude-trace.jsonl")):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            first = open(p, errors="replace").readline()
            kind = "codex" if ('"payload"' in first or '"session_meta"' in first) else "claude"
            traj = p; break
    cond = condense_codex if kind == "codex" else condense_claude
    t1 = cond(traj, 1) if traj else "(no trajectory file)"
    t2 = cond(traj, 2) if traj else ""
    open(os.path.join(b, "transcript_phase1.md"), "w").write(t1)
    if t2.strip(): open(os.path.join(b, "transcript_phase2.md"), "w").write(t2)
    turns_run = len([t for t in cell.get("turns", []) if t.get("verdict") in ("PASS", "FAIL")])
    open(os.path.join(b, "context.txt"), "w").write(
        f"model: {cell.get('model')} / {cell.get('effort')}\nrepo: {cell.get('task')}\n"
        f"graded turns: {turns_run}\n"
        "phase 2 (frozen turn, only run_train.py may change) applies to turn 2 and later.\n")
    log = git(ws, "log", "--reverse", "--format=%h %s")
    commits = [l.split()[0] for l in log.splitlines() if l.strip()]
    if len(commits) >= 2:
        open(os.path.join(b, "diff_phase1.stat"), "w").write(git(ws, "diff", "--stat", commits[0], commits[1]))
        patch = git(ws, "diff", commits[0], commits[1], "--", ".", ":(exclude)source_torch", ":(exclude)*.txt", ":(exclude)TPU_ACCESS.md")
        open(os.path.join(b, "diff_phase1.patch"), "w").write(cap(patch, CAP_PATCH))
    if len(commits) >= 3:
        ed = os.path.join(cell_dir, "entry_diff_turn2.patch")
        open(os.path.join(b, "diff_phase2.patch"), "w").write(open(ed).read() if os.path.exists(ed) else cap(git(ws, "diff", commits[1], commits[-1]), CAP_PATCH))
    rt = os.path.join(ws, "run_train.py")
    if os.path.exists(rt): open(os.path.join(b, "run_train.py"), "w").write(open(rt, errors="replace").read())
    fz = [t.get("verdict_detail", "") for t in cell.get("turns", []) if "FREEZE" in str(t.get("verdict_detail", "")) or "ENTRY-GROWTH" in str(t.get("verdict_detail", ""))]
    # Mechanical: which modules of the frozen tree does the graded entry import?
    # An entry that imports nothing local cannot be exercising the migrated code.
    loc = []
    if os.path.exists(rt):
        src = open(rt, errors="replace").read()
        mods = set((m.group(1) or m.group(2)).split(".")[0] for m in re.finditer(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import|^\s*import\s+([A-Za-z_][\w.]*)", src, re.M))
        loc = [x for x in sorted(mods) if x != "run_train" and (os.path.exists(os.path.join(ws, x + ".py")) or os.path.isdir(os.path.join(ws, x)))]
        fz.append(f"entry imports from the frozen tree: {', '.join(loc) if loc else 'NONE (ENTRY-NO-LOCAL-IMPORT: the entry cannot be using the port)'}")
    open(os.path.join(b, "freeze_check.txt"), "w").write("\n".join(fz) or "(no frozen turn, or freeze check clean: run_train.py only)")
    repo = (cell.get("task") or os.path.basename(cell_dir)).replace("_e2e", "")
    rtext = recipe_text(repo)
    open(os.path.join(b, "recipe.txt"), "w").write(rtext)   # v4: question 3 is judged against OUR recipe
    # The judges once had no source code in the bundle and asserted the source's learning-rate
    # warmup from memory (nanoGPT's older `it / warmup_iters`), failing two faithful ports whose formula matches the
    # vendored train.py. The bundle now carries the source files the recipe names, as vendored at the graded commit.
    src_files = source_files_for(repo, rtext, os.path.join(b, "source_files"))
    if src_files:
        with open(os.path.join(b, "context.txt"), "a") as fh:
            fh.write("source_files/: the source repository's own files that recipe.txt names (training script, data preparation, "
                     "evaluation), as vendored at the graded commit: " + ", ".join(src_files) + ". Rules about what the source does "
                     "(schedule, evaluation, splits, architecture) are judged against these files, not against memory of the repository.\n")
    crit = jload(os.path.join(REPO, f"tasks/{repo}/criteria.json"), {})
    metrics = {os.path.basename(p): jload(p, {}) for p in glob.glob(os.path.join(cell_dir, "grade_full_metrics*.json"))}
    summ = {"repo": repo, "model": cell.get("model"), "effort": cell.get("effort"), "verdict": cell.get("verdict"),
            "turns": [{k: t.get(k) for k in ("turn", "verdict", "verdict_detail", "weights_load", "runtime_s")} for t in cell.get("turns", [])],
            "bar": crit.get("bar"), "bar_cmp": crit.get("bar_cmp"), "metric_re": crit.get("metric_re"), "dataset_regex": crit.get("dataset_regex"),
            "dataset_note": compact(crit.get("dataset") or crit.get("bar_provenance")), "metrics_files": compact(metrics),
            "danger_flags": [open(p).read()[:1500] for p in glob.glob(os.path.join(cell_dir, "dangerous_turn*.txt"))]}
    json.dump(summ, open(os.path.join(b, "grade_summary.json"), "w"), indent=1, default=str)
    for gl in sorted(glob.glob(os.path.join(cell_dir, "grade_full_turn*.log")))[-2:]:
        try:
            lines = open(gl, errors="replace").read().splitlines()
            open(os.path.join(b, os.path.basename(gl).replace(".log", "_tail.txt")), "w").write("\n".join(lines[-80:]))
        except Exception: pass
    lines = []
    for src in (t1, t2):
        for m in URL_RE.finditer(src): lines.append(f"URL   {m.group(0)}")
        for l in src.splitlines():
            if l.startswith("$ ") and FETCH_RE.search(l): lines.append(f"FETCH {l[2:][:300]}")
    stat = open(os.path.join(b, "diff_phase1.stat")).read() if os.path.exists(os.path.join(b, "diff_phase1.stat")) else ""
    for l in stat.splitlines():
        f = l.split("|")[0].strip()
        if f and not f.startswith(("jax_", "run_train", "setup_env", "export_torch", "source_torch")) and "file" not in f: lines.append(f"WRITE {f}")
    open(os.path.join(b, "network_and_files.txt"), "w").write("\n".join(dict.fromkeys(lines)) or "(nothing extracted)")
    return b

# ---- the judge call (shared by judge_run.py and failure_judge.py) ---------------
def ensure_codex_home(model, effort):
    """A dedicated CODEX_HOME for the judges: auth copied from ~/.codex, model+effort pinned."""
    os.makedirs(JUDGE_HOME, exist_ok=True)
    auth = os.path.join(JUDGE_HOME, "auth.json")
    # Refresh the copy whenever ~/.codex/auth.json is newer: a stale copy holds a
    # spent refresh token ("refresh token was already used") and every judge run then
    # returns parse_error.
    src = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(src) and (not os.path.exists(auth) or os.path.getmtime(src) > os.path.getmtime(auth)):
        import shutil; shutil.copy(src, auth); os.chmod(auth, 0o600)
    open(os.path.join(JUDGE_HOME, "config.toml"), "w").write(f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n')
    return JUDGE_HOME

def run_judge(bundle, prompt_file, model, effort):
    home = ensure_codex_home(model, effort)
    out = os.path.join(bundle, "answer.txt")
    cmd = [CODEX_BIN, "-c", f'model_reasoning_effort="{effort}"', "--ask-for-approval", "never", "exec",
           "--cd", bundle, "--sandbox", "read-only", "--skip-git-repo-check", "--output-last-message", out, "--model", model, "-"]
    env = dict(os.environ, CODEX_HOME=home)
    r = subprocess.run(cmd, input=open(prompt_file).read(), capture_output=True, text=True, timeout=3600, env=env)
    txt = open(out).read() if os.path.exists(out) else r.stdout
    m = re.search(r"\{.*\}", txt, re.S)
    try: ans = json.loads(m.group(0)) if m else {"parse_error": txt[:2000], "rc": r.returncode, "stderr": r.stderr[-1500:]}
    except Exception: ans = {"parse_error": txt[:2000], "rc": r.returncode, "stderr": r.stderr[-1500:]}
    return ans, r.returncode

def main():
    ap = argparse.ArgumentParser(description="build the evidence bundle of one cell")
    ap.add_argument("cell_dir")
    ap.add_argument("--subdir", default="judge_trials/bundle", help="where under CELL_DIR to write it")
    a = ap.parse_args()
    b = build_bundle(a.cell_dir, subdir=a.subdir)
    size = sum(os.path.getsize(os.path.join(b, f)) for f in os.listdir(b)) // 1024
    print(f"bundle: {b} ({size} KB)")

if __name__ == "__main__":
    main()
