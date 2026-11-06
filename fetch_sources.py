#!/usr/bin/env python3
"""fetch_sources.py [REPO ...]

Put every task's PyTorch source at tasks/<repo>/source_torch/<dir>, exactly as the
benchmark vendored it: a shallow fetch of the pinned commit, then the paths in
task.yaml's source_drop are removed (assets, notebooks, unrelated subprojects the
benchmark never shipped), then the tree hash is compared with task.yaml's
source_tree_sha256. Needs git and network. With no argument, all tasks; a task
whose tree already matches is skipped.
"""
import hashlib, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP = {".git", "__pycache__"}


def read_task(path):
    t, drops, in_drop = {}, [], False
    for line in open(path):
        if in_drop and line.startswith("  - "):
            drops.append(line[4:].strip()); continue
        in_drop = False
        m = re.match(r"^([a-z0-9_]+):\s*(.*)$", line)
        if m:
            t[m.group(1)] = m.group(2).strip()
            if m.group(1) == "source_drop":
                in_drop = True
    t["source_drop"] = drops
    return t


def tree_hash(d):
    files = []
    for root, dirs, names in os.walk(d):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in names:
            if not f.endswith(".pyc"):
                p = os.path.join(root, f)
                files.append((os.path.relpath(p, d), p))
    h = hashlib.sha256()
    for rel, p in sorted(files):
        h.update(f"{rel}\0{hashlib.sha256(open(p, 'rb').read()).hexdigest()}\n".encode())
    return h.hexdigest(), len(files)


def main(argv):
    tasks = argv or sorted(d for d in os.listdir(os.path.join(ROOT, "tasks")) if os.path.exists(os.path.join(ROOT, "tasks", d, "task.yaml")))
    bad = 0
    for task in tasks:
        t = read_task(os.path.join(ROOT, "tasks", task, "task.yaml"))
        dest = os.path.join(ROOT, "tasks", task, t["source_dir"])
        if os.path.isdir(dest) and tree_hash(dest)[0] == t["source_tree_sha256"]:
            print(f"{task}: present"); continue
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
        subprocess.run(["git", "remote", "add", "origin", t["source_repo"]], cwd=dest, check=True)
        subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", t["source_commit"]], cwd=dest, check=True)
        subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest, check=True)
        shutil.rmtree(os.path.join(dest, ".git"))
        for rel in t["source_drop"]:
            p = os.path.join(dest, rel)
            if os.path.isdir(p): shutil.rmtree(p)
            elif os.path.exists(p): os.remove(p)
        got, n = tree_hash(dest)
        ok = got == t["source_tree_sha256"]
        bad += 0 if ok else 1
        print(f"{task}: {n} files, {'OK' if ok else 'HASH MISMATCH'} ({t['source_repo']} @ {t['source_commit'][:10]})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
