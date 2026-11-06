#!/usr/bin/env python3
"""The verdict comparator, called by the driver (run_cell.sh).
Usage:  compare_metric.py METRICS_JSON BAR DS_RE METRIC_RE CMP BAR2 [LOG_PATH]
Prints one line: "PASS ..." or "FAIL ...".

The rules:
  1. fields whose name is reference-like (reported/reference/published/readme/paper/expected/
     baseline/target/threshold/bar) are never graded;
  2. when several fields match, the headline split's field wins over val/best-so-far/running
     values;
  3. a value is rescaled between percent and fraction only when the field is a rate-like
     measure (accuracy, BLEU, F1, HR, NDCG, ...): a loss is never divided by 100;
  4. a band bar does not pick "the first value inside the band" among several fields: the
     field is chosen by rule 2 first, then compared.
"""
import json, math, re, sys

REF_RE = re.compile(r"reported|reference|published|readme|paper|expected|baseline|target|threshold|(^|_)bar(_|$)", re.I)
RATE_RE = re.compile(r"acc|accuracy|bleu|f1|hr@|hit|ndcg|rouge|prec|recall|auc|rate|score|top[0-9]|em($|_)|exact", re.I)

def compare(path, bar, ds_re, metric_re, cmp_, bar2, log_path=""):
    out = []
    def P(s): out.append(s)
    try:
        m = json.load(open(path))
    except Exception:
        return "FAIL run ok but no readable metrics JSON"
    cands, ds_names = [], []   # (path, value); candidacy is judged on the INNERMOST key (as before),
    # the full path only excludes reference-like / training-split parents and prefers test over val
    def named(kl):   # the candidacy predicate on a key name
        return bool(re.search(metric_re, kl)) if metric_re else ("val" in kl and "loss" in kl)
    def walk(o, key="", path="", in_list=False):
        if isinstance(o, dict):
            for kk, v in o.items():
                # a metric-named key whose value is a
                # dict with a numeric `mean` or `value` leaf counts as that metric, under the metric's
                # own path (the innermost-key rule would otherwise see only `mean`). Lists stay excluded.
                if isinstance(v, dict) and not in_list and named(kk.lower()):
                    for agg in ("mean", "value"):
                        a = v.get(agg)
                        if isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a):
                            cands.append(((f"{path}/{kk}" if path else kk).lower(), float(a)))
                            break
                walk(v, kk, (f"{path}/{kk}" if path else kk), in_list)
        elif isinstance(o, list):
            for v in o:
                walk(v, key, path, True)
        else:
            if in_list:
                return
            kl = key.lower()
            if (isinstance(o, (int, float)) and not isinstance(o, bool)
                    and math.isfinite(o)
                    and (re.search(metric_re, kl) if metric_re
                         else ("val" in kl and "loss" in kl))):
                cands.append((path.lower(), float(o)))
            elif isinstance(o, str) and ("dataset" in kl or kl == "data"):
                ds_names.append(o)
    walk(m)
    kept = [(k, v) for k, v in cands if not REF_RE.search(k)]          # rule 1 (path-wide)
    cands = kept
    def alternatives(rx):
        parts, depth, cur, esc, cls = [], 0, "", False, False
        for ch in rx:
            if esc: cur += ch; esc = False; continue
            if ch == "\\": cur += ch; esc = True; continue
            if cls: cur += ch; cls = ch != "]"; continue
            if ch == "[": cur += ch; cls = True; continue
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            if ch == "|" and depth == 0: parts.append(cur); cur = ""; continue
            cur += ch
        parts.append(cur); return [x for x in parts if x.strip()] or [rx]
    inner = lambda k: k.rsplit("/", 1)[-1]
    if cands and metric_re:
        alts = alternatives(metric_re)                                    # rule 2: regex order, on the innermost key
        def first_alt(k):
            for i, a in enumerate(alts):
                try:
                    if re.search(a, inner(k)): return i
                except re.error: pass
            return len(alts)
        best_i = min(first_alt(k) for k, _ in cands)
        cands = [(k, v) for k, v in cands if first_alt(k) == best_i]
        if "test" in metric_re.lower():                                   # test over val, judged on the path
            tk = [(k, v) for k, v in cands if "test" in k]
            if tk and any("val" in k and "test" not in k for k, _ in cands): cands = tk
        nt = [(k, v) for k, v in cands if not re.search(r"(^|[^a-z])train(ing)?([^a-z]|$)", k)]   # never a training-split value
        if nt and len(nt) < len(cands): cands = nt
    vals = [v for _, v in cands]
    suffix = ""
    if ds_re:
        if ds_names and not any(re.search(ds_re, d, re.I) for d in ds_names):
            return f"FAIL trained on {'/'.join(ds_names[:3])!r}, not the headline dataset"
        if ds_names:
            suffix = f" (dataset: {ds_names[0]})"
        else:
            hit = ""
            try:
                with open(log_path, "rb") as fh:
                    text = fh.read().decode("utf-8", "replace")
                mm = re.search(ds_re, text, re.I)
                hit = mm.group(0) if mm else ""
            except Exception:
                hit = ""
            if not hit:
                return "FAIL neither the metrics JSON nor the run log names the headline dataset (required by the goal)"
            suffix = f" (dataset evidence from run log: {hit!r})"
    if cands:
        suffix += f" [field: {cands[0][0]}]" if len(set(k for k, _ in cands)) == 1 else f" [fields: {','.join(sorted(set(k for k, _ in cands)))}]"
    if not vals:
        return "FAIL no reported graded-metric field in metrics JSON"
    if cmp_ == "ge":
        v = max(vals)
        return (f"PASS max reported metric {v:.6f} >= {bar}{suffix}") if v >= bar else (f"FAIL max reported metric {v:.6f} < {bar}{suffix}")
    if cmp_ == "band":
        hi = float(bar2)
        ok = [v for v in vals if bar <= v <= hi]
        v = ok[0] if ok else vals[0]
        ratelike = any(RATE_RE.search(k) for k, _ in cands)   # rule 3
        resc = []
        if ratelike:
            for x in vals:
                if hi <= 1.0 and x > hi and bar <= x / 100.0 <= hi:
                    resc.append((x / 100.0, x, "/100"))
                elif hi == 100.0 and 0.0 < x <= 1.0 and bar <= x * 100.0 <= hi:
                    resc.append((x * 100.0, x, "x100"))
        above = [x for x in vals if x > hi]
        if ok:
            return f"PASS metric {v:.6f} within band [{bar}, {hi}]{suffix}"
        if resc:
            r, x, how = resc[0]
            return f"PASS metric {r:.6f} (reported {x:.6f}, rescaled {how}: other unit convention) within band [{bar}, {hi}]{suffix}"
        if above and ratelike and any(x > 100.0 * hi for x in above):
            return f"FAIL UNIT-EXCEPTION: metric {max(above):.6f} far above the ceiling {hi} of band [{bar}, {hi}] (unrecognised scale){suffix}"
        if above and ratelike:
            r = max(above) / 100.0
            return f"FAIL metric {r:.6f} (reported {max(above):.6f}, rescaled /100) outside band [{bar}, {hi}]{suffix}"
        if above:
            return f"FAIL metric {max(above):.6f} outside band [{bar}, {hi}]{suffix}"
        return f"FAIL metric {vals[0]:.6f} outside band [{bar}, {hi}]{suffix}"
    v = min(vals)
    return (f"PASS min reported metric {v:.6f} <= {bar}{suffix}") if v <= bar else (f"FAIL min reported metric {v:.6f} > {bar}{suffix}")

if __name__ == "__main__":
    a = sys.argv
    path, bar, ds_re, metric_re, cmp_, bar2 = a[1], float(a[2]), a[3], a[4], a[5], a[6]
    log_path = a[7] if len(a) > 7 else ""
    print(compare(path, bar, ds_re, metric_re, cmp_, bar2, log_path))
