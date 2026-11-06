#!/usr/bin/env python3
"""Stage 2 of TPU grading: the port's exported weights must load name-for-name with
exact shapes into the UNMODIFIED source PyTorch model.

    check_weight_load.py <repo> <npz> [--json OUT]      grade one export
    check_weight_load.py <repo|all> --summary           print the source state_dict summary
    check_weight_load.py <repo|all> --self-test         round-trip + negative tests
    check_weight_load.py --list                         registry entries

Exit code: 0 PASS, 1 FAIL, 2 no-npz / source-model build error, 3 stage 2 not applicable
(a registry entry that says so, or no registry entry at all). The ``summary`` field of
the JSON report is the ledger ``weights_load`` value: ``PASS``,
``FAIL:<n> missing, <m> extra, <k> shape``, ``no-npz``, ``n/a: <reason>`` or
``error:<what>``.

Comparison rules (documented normalisations, applied to BOTH sides):
  * leading ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) prefixes are stripped,
    repeatedly, plus any per-entry prefixes (x_transformers' ``net.`` wrapper); a
    ``_orig_mod`` path component anywhere is dropped;
  * exempt deterministic buffers (registry, plus ``num_batches_tracked`` for every task
    BatchNorm counters) may be missing or extra, shape-checked if present on both sides;
  * optional dead parameters (registry; AST's unused DeiT heads, tgn's memory state) may
    be missing, shape-checked if present, recorded under ``optional_missing``;
  * a tie group (registry) is satisfied by any one member with the right shape; members
    that are both present must be exactly equal unless the entry says tied-or-untied is
    fine; a group whose members are all optional may be absent;
  * ``alt_shapes`` (registry) admits an alternative shape for a key, right-padded on load;
  * entries with ``build_from_npz`` construct the source model from metadata carried in
    the export itself (tensorf's data-dependent grid), so the shape check is an
    internal-consistency check there;
  * dtypes are not compared (numpy has no bf16); values are cast to the model dtype.
The final step really loads: every state_dict tensor is overwritten in place from the
npz (exact shape asserted), one array at a time so peak memory is about the model alone.

Run with a python that has torch (XSTACK_TORCH_PYTHON; CPU is enough, no CUDA is touched).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_load_registry import REGISTRY, RepoEntry  # noqa: E402

STRIP_PREFIXES = ("module.", "_orig_mod.")
# BatchNorm's int64 step counter: every BN-carrying source has it, no Flax port does.
FLEET_EXEMPT = (r".*num_batches_tracked",)
Shape = tuple[int, ...]

EXIT_PASS, EXIT_FAIL, EXIT_ERROR, EXIT_NA = 0, 1, 2, 3


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------


def normalise_key(key: str, extra_prefixes: tuple[str, ...] = ()) -> str:
    prefixes = STRIP_PREFIXES + tuple(extra_prefixes)
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return ".".join(p for p in key.split(".") if p != "_orig_mod")


def normalise_keys(
    keys: list[str], extra_prefixes: tuple[str, ...] = ()
) -> tuple[dict[str, str], list[list[str]]]:
    """Return ``{normalised: original}`` and the groups of originals that collide."""
    out: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for k in keys:
        n = normalise_key(k, extra_prefixes)
        if n in out:
            collisions.setdefault(n, [out[n]]).append(k)
        else:
            out[n] = k
    return out, list(collisions.values())


# --------------------------------------------------------------------------
# streaming npz access (an export can be GBs; never hold it all)
# --------------------------------------------------------------------------


class NpzView:
    """Read-only view of an ``np.savez`` archive: shapes from the .npy headers, arrays on
    demand. ``allow_pickle`` is always False."""

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self._members = {
            n[: -len(".npy")]: n for n in self._zip.namelist() if n.endswith(".npy")
        }
        self._headers: dict[str, tuple[Shape, np.dtype]] = {}

    def close(self) -> None:
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def keys(self) -> list[str]:
        return list(self._members)

    def __contains__(self, key: str) -> bool:
        return key in self._members

    def header(self, key: str) -> tuple[Shape, np.dtype]:
        if key not in self._headers:
            with self._zip.open(self._members[key]) as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
            self._headers[key] = (tuple(shape), dtype)
        return self._headers[key]

    def shape(self, key: str) -> Shape:
        return self.header(key)[0]

    def array(self, key: str) -> np.ndarray:
        with self._zip.open(self._members[key]) as fh:
            return np.lib.format.read_array(fh, allow_pickle=False)


def npz_to_tensor(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Cast an npz array to the model tensor's dtype. Handles ml_dtypes bfloat16 without
    importing ml_dtypes (bit-reinterpret) and refuses object arrays."""
    if arr.dtype == object:
        raise TypeError("object array")
    if arr.dtype.name == "bfloat16":
        bits = np.ascontiguousarray(arr).view(np.int16)
        t = torch.from_numpy(bits).view(torch.bfloat16)
    else:
        t = torch.from_numpy(np.ascontiguousarray(arr))
    # np.ascontiguousarray promotes a 0-d array to shape (1,): restore the exact shape
    # (BatchNorm counters, tdmpc2's log_std scalars)
    return t.reshape(arr.shape).to(like.dtype)


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------


def _fmt(k: str, t: torch.Tensor) -> str:
    return f"{k} {tuple(t.shape)} {str(t.dtype)[len('torch.'):]}"


def tensor_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """state_dict() minus non-tensor entries (tensordict 0.8's TensorDictParams put a
    torch.Size and a device string under *.__batch_size / *.__device; nothing loads there)."""
    return {k: v for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}


def state_dict_summary(model: torch.nn.Module, n_show: int = 5) -> dict[str, Any]:
    sd = tensor_state_dict(model)
    keys = list(sd)
    params = dict(model.named_parameters())
    buffer_names = {n for n, _ in model.named_buffers()}
    n_buffers = sum(1 for k in keys if k in buffer_names)  # persistent ones only
    unique = {p.data_ptr(): p.numel() for p in params.values()}
    return {
        "n_tensors": len(sd),
        "n_parameters": len(params),
        "n_buffers": n_buffers,
        "n_other": len(sd) - len(params) - n_buffers,  # tied aliases, TensorDict metadata
        "state_dict_numel": int(sum(t.numel() for t in sd.values())),
        "unique_parameter_numel": int(sum(unique.values())),
        "first_keys": [_fmt(k, sd[k]) for k in keys[:n_show]],
        "last_keys": [_fmt(k, sd[k]) for k in keys[-n_show:]],
    }


def _match_set(patterns: tuple[str, ...], keys: list[str]) -> set[str]:
    pats = [re.compile(p) for p in patterns]
    return {k for k in keys if any(p.fullmatch(k) for p in pats)}


def exempt_patterns(entry: RepoEntry) -> tuple[str, ...]:
    return FLEET_EXEMPT + tuple(entry.exempt_buffers)


def _accepted_shapes(entry: RepoEntry, key: str, model_shape: Shape) -> set[Shape]:
    return {model_shape, *entry.alt_shapes.get(key, ())}


def _fit(entry: RepoEntry, key: str, t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Right-pad an accepted alternative 1-D shape up to the model's shape."""
    if tuple(t.shape) == tuple(target.shape):
        return t
    if t.dim() != 1 or t.numel() > target.numel():
        raise ValueError(f"{key}: cannot pad {tuple(t.shape)} to {tuple(target.shape)}")
    pad = torch.full((target.numel() - t.numel(),), entry.pad_value, dtype=t.dtype)
    return torch.cat([t, pad])


def compare(entry: RepoEntry, model: torch.nn.Module, npz: NpzView) -> dict[str, Any]:
    """Compare the model's state_dict against the npz; on a clean diff, load in place."""
    strip = tuple(entry.extra_strip_prefixes)
    sd = tensor_state_dict(model)
    model_norm, model_coll = normalise_keys(list(sd), strip)
    npz_norm, npz_coll = normalise_keys(npz.keys(), strip)
    model_keys = set(model_norm)
    npz_keys = set(npz_norm)

    exempt_model = _match_set(exempt_patterns(entry), sorted(model_keys))
    exempt_npz = _match_set(exempt_patterns(entry), sorted(npz_keys))
    optional_model = _match_set(entry.optional_keys, sorted(model_keys))
    # sanity: an exempt pattern must never swallow a learned parameter
    param_names = {normalise_key(n, strip) for n, _ in model.named_parameters()}
    exempt_params = sorted(exempt_model & param_names)

    tie_members = {m for g in entry.tie_groups for m in g}

    missing: list[str] = []
    extra: list[str] = []
    shape_bad: list[dict[str, Any]] = []
    exempt_missing: list[str] = []
    exempt_extra: list[str] = []
    optional_missing: list[str] = []
    matched: dict[str, str] = {}  # normalised model key -> npz original key that fills it

    for k in sorted(model_keys):
        mshape = tuple(sd[model_norm[k]].shape)
        if k in npz_keys:
            got = npz.shape(npz_norm[k])
            if got in _accepted_shapes(entry, k, mshape):
                matched[k] = npz_norm[k]
            else:
                shape_bad.append({"key": k, "expected": list(mshape), "got": list(got)})
        elif k in exempt_model:
            exempt_missing.append(k)
        elif k in optional_model:
            optional_missing.append(k)
        elif k in tie_members:
            continue  # resolved below
        else:
            missing.append(k)

    for k in sorted(npz_keys - model_keys):
        (exempt_extra if k in exempt_npz else extra).append(k)

    tie_reports: list[dict[str, Any]] = []
    tie_value_bad: list[str] = []
    for group in entry.tie_groups:
        present = [m for m in group if m in npz_keys]
        ok = [m for m in present if m in matched]
        rep: dict[str, Any] = {
            "group": list(group), "present_in_npz": present, "satisfied": bool(ok),
        }
        if not ok:
            if not present and all(m in optional_model for m in group):
                rep["satisfied"] = "optional"  # e.g. tgn memory state kept outside the tree
            elif not present:  # mis-shaped members are already under shape_bad
                missing.append("|".join(group))
        else:
            if len(present) > 1 and not entry.tied_or_untied_ok:
                ref = npz.array(npz_norm[ok[0]])
                for m in present[1:]:
                    other = npz.array(npz_norm[m])
                    if other.shape != ref.shape or not np.array_equal(
                        other.astype(np.float64), ref.astype(np.float64)
                    ):
                        tie_value_bad.append(f"{ok[0]} != {m}")
                del ref, other
                rep["equal"] = not tie_value_bad
            elif len(present) > 1:
                rep["equal"] = "not required (tied or untied both sanctioned)"
            for m in group:  # every alias is filled from the satisfied member
                matched.setdefault(m, npz_norm[ok[0]])
        tie_reports.append(rep)

    duplicates = model_coll + npz_coll
    fail = bool(missing or extra or shape_bad or tie_value_bad or duplicates or exempt_params)

    # the load itself: in place, one array at a time, exact shape asserted before copy_
    loaded = False
    load_error: str | None = None
    alt_used: list[str] = []
    if not fail:
        try:
            with torch.no_grad():
                for k, mk in model_norm.items():
                    if k not in matched:
                        continue  # exempt/optional key absent from the npz keeps the source value
                    target = sd[mk]
                    t = npz_to_tensor(npz.array(matched[k]), target)
                    if tuple(t.shape) != tuple(target.shape):
                        alt_used.append(k)
                        t = _fit(entry, k, t, target)
                    if tuple(t.shape) != tuple(target.shape):  # never rely on broadcasting
                        raise ValueError(f"{k}: {tuple(t.shape)} vs {tuple(target.shape)}")
                    target.copy_(t)
            loaded = True
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            load_error = f"{type(exc).__name__}: {exc}"
            fail = True

    n_missing, n_extra, n_shape = len(missing), len(extra), len(shape_bad)
    summary = f"FAIL:{n_missing} missing, {n_extra} extra, {n_shape} shape" if fail else "PASS"
    return {
        "verdict": "FAIL" if fail else "PASS",
        "summary": summary,
        "counts": {
            "model_tensors": len(sd),
            "npz_arrays": len(npz_keys),
            "matched": len(matched),
            "missing": n_missing,
            "extra": n_extra,
            "shape_mismatch": n_shape,
            "exempt_missing": len(exempt_missing),
            "exempt_extra": len(exempt_extra),
            "optional_missing": len(optional_missing),
            "tie_value_mismatch": len(tie_value_bad),
            "duplicate_after_normalisation": len(duplicates),
        },
        "missing": missing,
        "extra": extra,
        "shape_mismatch": shape_bad,
        "exempt_buffers": sorted(exempt_model),
        "exempt_missing": exempt_missing,
        "exempt_extra": exempt_extra,
        "exempt_patterns_matching_parameters": exempt_params,
        "optional_keys": sorted(optional_model),
        "optional_missing": optional_missing,
        "tie_groups": tie_reports,
        "tie_value_mismatch": tie_value_bad,
        "duplicate_after_normalisation": duplicates,
        "alt_shapes_used": sorted(alt_used),
        "loaded_into_source_model": loaded,
        "load_error": load_error,
        "normalisation": {
            "stripped_prefixes": list(STRIP_PREFIXES + strip),
            "dropped_components": ["_orig_mod"],
        },
    }


def build_model(entry: RepoEntry, npz: NpzView | None = None) -> torch.nn.Module:
    if entry.not_applicable:
        raise RuntimeError(f"{entry.repo}: stage 2 not applicable ({entry.not_applicable})")
    if entry.build is None and entry.build_from_npz is None:
        raise RuntimeError(f"{entry.repo}: stub entry ({entry.stub_reason})")
    torch.manual_seed(0)
    t0 = time.time()
    model = entry.build_from_npz(npz) if entry.build_from_npz is not None else entry.build()
    model.eval()
    print(f"[{entry.repo}] built source model in {time.time() - t0:.1f}s", file=sys.stderr)
    return model


def run_check(repo: str, npz_path: Path, json_out: Path | None) -> int:
    entry = REGISTRY.get(repo)
    report: dict[str, Any] = {
        "repo": repo,
        "npz": str(npz_path),
        "source_dir": entry.source_dir if entry else None,
        "config": entry.config_note if entry else None,
        "torch": torch.__version__,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if entry is None:
        report.update(verdict="n/a", summary=f"n/a: no registry entry for {repo}")
        _emit(report, json_out)
        return EXIT_NA
    if entry.not_applicable:
        report.update(verdict="n/a", summary=f"n/a: {entry.not_applicable}")
        _emit(report, json_out)
        return EXIT_NA
    if not npz_path.is_file():
        report.update(verdict="no-npz", summary="no-npz")
        _emit(report, json_out)
        return EXIT_ERROR
    try:
        npz = NpzView(npz_path)
    except Exception as exc:  # noqa: BLE001 - a corrupt export is the port's FAIL
        report.update(
            verdict="FAIL", summary="FAIL:unreadable npz",
            load_error=f"{type(exc).__name__}: {exc}",
        )
        _emit(report, json_out)
        return EXIT_FAIL
    with npz:
        try:
            model = build_model(entry, npz)
        except Exception as exc:  # noqa: BLE001 - grader-side problem, distinct exit code
            report.update(
                verdict="error", summary="error:source model build failed",
                load_error=f"{type(exc).__name__}: {exc}",
            )
            _emit(report, json_out)
            return EXIT_ERROR
        note = getattr(model, "_stage2_note", None)
        if note:
            report["config"] = f"{report['config']}; {note}"
        report["source_state_dict"] = state_dict_summary(model)
        report.update(compare(entry, model, npz))
    _emit(report, json_out)
    return EXIT_PASS if report["verdict"] == "PASS" else EXIT_FAIL


def _emit(report: dict[str, Any], json_out: Path | None) -> None:
    c = report.get("counts")
    print(f"{report['repo']}: {report['summary']}")
    if c:
        print(
            f"  model tensors {c['model_tensors']}, npz arrays {c['npz_arrays']}, "
            f"matched {c['matched']}, exempt missing {c['exempt_missing']}, "
            f"exempt extra {c['exempt_extra']}, optional missing {c['optional_missing']}"
        )
        for name in (
            "missing", "extra", "tie_value_mismatch", "duplicate_after_normalisation",
            "exempt_patterns_matching_parameters",
        ):
            if report[name]:
                more = " ..." if len(report[name]) > 10 else ""
                print(f"  {name} ({len(report[name])}): {report[name][:10]}{more}")
        if report["shape_mismatch"]:
            shown = [
                f"{d['key']} expected {d['expected']} got {d['got']}"
                for d in report["shape_mismatch"][:10]
            ]
            print(f"  shape_mismatch ({len(report['shape_mismatch'])}): {shown}")
        if report["alt_shapes_used"]:
            print(f"  alt shapes accepted: {report['alt_shapes_used']}")
        print(f"  loaded into source model: {report['loaded_into_source_model']}")
    if report.get("load_error"):
        print(f"  error: {report['load_error']}")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  report -> {json_out}")


# --------------------------------------------------------------------------
# summary / self-test
# --------------------------------------------------------------------------


def _get(repo: str) -> RepoEntry:
    try:
        return REGISTRY[repo]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise SystemExit(f"unknown repo {repo!r}; registry has: {known}") from None


def print_summary(repo: str) -> None:
    entry = _get(repo)
    print(f"=== {repo}: {entry.config_note}")
    print(f"    source: {entry.source_dir}")
    if entry.not_applicable:
        print(f"    N/A: {entry.not_applicable}")
        return
    if entry.build is None and entry.build_from_npz is None:
        print(f"    STUB: {entry.stub_reason}")
        return
    model = build_model(entry)
    s = state_dict_summary(model)
    print(
        f"    state_dict tensors {s['n_tensors']} (parameters {s['n_parameters']}, "
        f"persistent buffers {s['n_buffers']}, other/aliases {s['n_other']}); "
        f"state_dict numel {s['state_dict_numel']:,}; "
        f"unique parameters {s['unique_parameter_numel']:,}"
    )
    print("    first:", *s["first_keys"], sep="\n      ")
    print("    last:", *s["last_keys"], sep="\n      ")
    keys = list(tensor_state_dict(model))
    ex = sorted(_match_set(exempt_patterns(entry), keys))
    if ex:
        print(f"    exempt buffers matched: {ex}")
    if entry.optional_keys:
        print(f"    optional (dead) keys matched: {sorted(_match_set(entry.optional_keys, keys))}")
    if entry.tie_groups:
        print(f"    tie groups: {entry.tie_groups}"
              + (" (tied or untied both accepted)" if entry.tied_or_untied_ok else ""))
    if entry.alt_shapes:
        print(f"    alt shapes: {entry.alt_shapes}")
    if entry.extra_strip_prefixes:
        print(f"    extra stripped prefixes: {entry.extra_strip_prefixes}")
    note = getattr(model, "_stage2_note", None)
    if note:
        print(f"    note: {note}")


def _dump(sd: dict[str, torch.Tensor], path: Path, transform=None) -> None:
    """Write the state_dict as the port would: numpy arrays keyed by source names. bf16
    tensors go out as float32 (numpy has no bf16). ``transform`` edits the dict first."""
    arrays: dict[str, np.ndarray] = {}
    for k, t in sd.items():
        a = t.detach().cpu()
        arrays[k] = a.float().numpy() if a.dtype == torch.bfloat16 else a.numpy()
    if transform is not None:
        arrays = transform(arrays)
    np.savez(path, **arrays)


def self_test(repo: str, tmp_root: Path | None) -> bool:
    entry = _get(repo)
    if entry.not_applicable:
        print(f"[{repo}] SKIP self-test: stage 2 not applicable ({entry.not_applicable})")
        return True
    if entry.build is None and entry.build_from_npz is None:
        print(f"[{repo}] SKIP self-test: stub entry ({entry.stub_reason})")
        return True
    model = build_model(entry)
    sd = tensor_state_dict(model)
    exempt = _match_set(exempt_patterns(entry), list(sd))
    optional = _match_set(entry.optional_keys, list(sd))
    ties = {m for g in entry.tie_groups for m in g}
    skip = exempt | optional | ties  # never pick these as the victim of a negative case
    results: list[tuple[str, bool, str]] = []

    def case(name: str, transform, expect: str, want=None) -> None:
        with tempfile.TemporaryDirectory(dir=tmp_root) as d:
            p = Path(d) / f"{repo}_{name}.npz"
            _dump(sd, p, transform)
            with NpzView(p) as npz:
                m = build_model(entry, npz) if entry.build_from_npz is not None else model
                rep = compare(entry, m, npz)
        ok = rep["verdict"] == expect and (want is None or want(rep))
        results.append((name, ok, rep["summary"]))
        print(f"  [{'ok' if ok else 'BAD'}] {name}: expect {expect}, got {rep['summary']}")
        if not ok:
            for f in ("load_error", "tie_value_mismatch", "duplicate_after_normalisation",
                      "exempt_patterns_matching_parameters", "missing", "extra", "shape_mismatch"):
                if rep.get(f):
                    print(f"        {f}: {str(rep[f])[:300]}")

    def without(*drop: str):
        return lambda a: {k: v for k, v in a.items() if k not in drop}

    # 1. exact round trip, really loaded
    case("roundtrip", None, "PASS", lambda r: r["loaded_into_source_model"])
    # 2. DDP + compile prefixes on every key (+ the entry's own wrapper prefix)
    wrap = "module._orig_mod." + "".join(entry.extra_strip_prefixes)
    case("ddp_compile_prefix", lambda a: {wrap + k: v for k, v in a.items()}, "PASS")
    # 3. exempt buffers omitted from the export
    if exempt:
        case("exempt_omitted", without(*exempt), "PASS",
             lambda r: r["counts"]["exempt_missing"] == len(exempt))
    # 4. ties: either alias alone suffices; both present but unequal fails unless sanctioned
    for group in entry.tie_groups:
        if all(m in optional for m in group):
            continue  # covered by the optional cases below
        for drop in group:
            case(f"tie_drop_{drop}", without(drop), "PASS")

        def untie(a, f=group[0]):
            return {**a, f: a[f] + 1.0}

        if entry.tied_or_untied_ok:
            case("tie_unequal_sanctioned", untie, "PASS")
        else:
            case("tie_unequal", untie, "FAIL", lambda r: r["counts"]["tie_value_mismatch"] >= 1)
    # 4b. dead parameters omitted from the export -> PASS, recorded
    if optional:
        case("optional_omitted", without(*optional), "PASS",
             lambda r: r["counts"]["optional_missing"] == len(optional))
        bad_opt = next(
            (k for k in sorted(optional) if sd[k].dim() >= 1 and sd[k].shape[0] > 1), None
        )
        if bad_opt is not None:  # 0-d optional keys (patchtst's attention scale) have no dim to cut
            case(f"truncate_optional_{bad_opt}", lambda a: {**a, bad_opt: a[bad_opt][:-1]},
                 "FAIL", lambda r: [d["key"] for d in r["shape_mismatch"]] == [bad_opt])
    # 5. drop one real parameter -> FAIL naming it
    victim = next(k for k in sd if k not in skip)
    case(f"drop_{victim}", without(victim), "FAIL", lambda r: r["missing"] == [victim])
    # 6. truncate one parameter's leading dim -> FAIL with a shape entry
    victim2 = next(k for k in sd if k not in skip and sd[k].dim() >= 1 and sd[k].shape[0] > 1)
    case(f"truncate_{victim2}", lambda a: {**a, victim2: a[victim2][:-1]}, "FAIL",
         lambda r: [d["key"] for d in r["shape_mismatch"]] == [victim2])
    # 7. an unknown extra tensor -> FAIL
    case("extra_key", lambda a: {**a, "not_in_source.weight": np.zeros((2, 2), np.float32)},
         "FAIL", lambda r: r["extra"] == ["not_in_source.weight"])
    # 8. alt shapes (modded scalars at world size 4) accepted; other lengths not
    for key, shapes in entry.alt_shapes.items():
        for shp in shapes:
            case(f"alt_shape_{key}_{list(shp)}",
                 lambda a, key=key, shp=shp: {**a, key: np.ones(shp, dtype=a[key].dtype)},
                 "PASS", lambda r, key=key: r["alt_shapes_used"] == [key])
            case(f"bad_shape_{key}",
                 lambda a, key=key, shp=shp: {**a, key: np.ones((shp[0] - 6,), a[key].dtype)},
                 "FAIL", lambda r, key=key: r["shape_mismatch"][0]["key"] == key)
    # 9. a float16 export of a bf16 tensor still loads (dtype is not compared)
    bf16_key = next((k for k, t in sd.items() if t.dtype == torch.bfloat16), None)
    if bf16_key is not None:
        case(f"fp16_export_{bf16_key}",
             lambda a, k=bf16_key: {**a, k: a[k].astype(np.float16)}, "PASS")
    # 10. export-driven construction: the entry's own extra self-test cases
    for name, transform, expect in entry.self_test_cases(sd):
        case(name, transform, expect)

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"[{repo}] self-test: {n_ok}/{len(results)} cases as expected")
    return n_ok == len(results)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("repo", nargs="?", help="registry key, or 'all' for --summary/--self-test")
    ap.add_argument("npz", nargs="?", help="the port's exported weights (grade_weights.npz)")
    ap.add_argument("--json", type=Path, help="write the report here")
    ap.add_argument("--summary", action="store_true",
                    help="build the source model, print its state_dict summary")
    ap.add_argument("--self-test", action="store_true",
                    help="round-trip and negative tests, no npz needed")
    ap.add_argument("--list", action="store_true", help="list registry entries")
    ap.add_argument("--tmpdir", type=Path, default=None,
                    help="where --self-test writes its temporary npz files")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    torch.set_grad_enabled(False)

    if args.list:
        for name, e in REGISTRY.items():
            if e.not_applicable:
                tag = "N/A"
            elif e.build is None and e.build_from_npz is None:
                tag = "STUB"
            else:
                tag = "ok"
            print(f"{name:26s} {tag:5s} {e.config_note}")
        return 0
    if args.repo is None:
        ap.error("repo is required")
    repos = list(REGISTRY) if args.repo == "all" else [args.repo]
    if args.summary:
        for r in repos:
            print_summary(r)
        return 0
    if args.self_test:
        outcomes = [self_test(r, args.tmpdir) for r in repos]  # a list: run every repo
        print("SELF-TEST", "PASS" if all(outcomes) else "FAIL")
        return 0 if all(outcomes) else 1
    if args.npz is None:
        ap.error("npz is required unless --summary/--self-test/--list")
    if args.repo == "all":
        ap.error("'all' is only for --summary/--self-test")
    return run_check(args.repo, Path(args.npz), args.json)


if __name__ == "__main__":
    sys.exit(main())
