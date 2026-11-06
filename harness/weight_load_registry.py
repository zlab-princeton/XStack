"""Stage-2 (weight load) registry: how to build each repo's UNMODIFIED source
PyTorch model at its canonical graded configuration, on CPU, from the vendored
tree under ``tasks/<repo>/source_torch/``.

One ``RepoEntry`` per task. ``check_weight_load.py`` is the consumer: it
builds the model, takes ``model.state_dict()`` and compares it name-for-name and
shape-for-shape against the port's exported ``grade_weights.npz``.

Conventions:

- ``exempt_buffers`` are regex patterns (``re.fullmatch`` against state_dict
  keys) naming deterministic, non-learned buffers a port need not export
  (causal-mask ``attn.bias``, rotary caches, ...). They may be missing from
  the npz and may appear in the npz even when the source model has none (the
  source's own presence can depend on the torch version, e.g. nanogpt's
  flash-attention branch). When present on both sides they are still
  shape-checked. The checker adds ``num_batches_tracked`` for every task.
- ``optional_keys`` are regex patterns for parameters the source constructs
  but a port may legitimately omit (AST's unused DeiT heads, tgn's
  requires_grad=False memory state, PatchTST's constant attention scale):
  absent is recorded, not failed; present is shape-checked.
- ``tie_groups`` are sets of state_dict keys that alias one storage in the
  source (weight tying). A group is satisfied when at least one member is in
  the npz with the right shape; if several are present they must be equal,
  unless ``tied_or_untied_ok`` (word_language_model: the band sanctions both).
  A group whose members are all optional may be entirely absent.
- ``alt_shapes`` maps a key to alternative shapes accepted in the npz; a 1-D
  array shorter than the model's is right-padded with ``pad_value`` at load
  (modded_nanogpt's ``scalars`` at world size 4 vs 8).
- ``extra_strip_prefixes`` are wrapper prefixes stripped from both sides in
  addition to ``module.`` / ``_orig_mod.`` (x_transformers' ``net.``).
- ``build_from_npz`` replaces ``build`` when the constructor needs metadata
  from the export itself (tensorf's data-dependent grid); it receives the
  NpzView (or None for --summary/--self-test) and may set ``_stage2_note``.
- ``not_applicable`` marks repos where stage 2 has no meaning (inference
  engines, trajectory-dependent pruned shapes): the checker returns
  ``n/a: <reason>`` with exit code 3.

Everything imports by path from the vendored tree; nothing here depends on a
pip-installed copy of any source repo.
Third-party libraries a source needs at construction time live in separate
``uv pip install --target`` dirs (dependency sites) (``EXTRA_SITES``,
override each with ``XSTACK_TORCH_SITE_<NAME>``) so the shared torch venv is
never mutated. Model construction is lazy, so importing this module costs
nothing beyond torch.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

_SITES = Path(__import__("os").environ.get("XSTACK_WORK", "work")) / "torch_sites"   # per-library site-packages for the weight-load check
# name -> default --target dir of the dependency site (one per family, never the venv)
EXTRA_SITES: dict[str, Path] = {
    # timm==0.4.5 (+ pillow, torchvision 0.28 = torch 2.13's pair, wget): ast, airbench
    "timm045": _SITES / "timm045",
    # transformers 4.x (+ huggingface_hub, tokenizers, safetensors, regex): simcse
    "transformers": _SITES / "transformers",
    # pytorch_lightning, omegaconf, hydra-core, torchmetrics, Levenshtein, six: deepspeech
    "lightning": _SITES / "lightning",
    # tensordict==0.8.3 (the source's pin; imports on torch 2.13): tdmpc2. Newer tensordict
    # changes the state_dict layout (zero-storage *.module.* templates instead of the
    # __batch_size/__device metadata entries); both layouts are handled by the entry.
    "tensordict": _SITES / "tensordict",
    # scipy, scikit-learn (+ joblib, threadpoolctl): tabr
    "scientific": _SITES / "scientific",
    # faiss-cpu, delu, tomli, tomli_w: tabr
    "tabr": _SITES / "tabr",
    # tensorboard (+ absl, grpcio, markdown, protobuf, werkzeug): makemore and tabr import
    # torch.utils.tensorboard at module level
    "tensorboard": _SITES / "tensorboard",
}


def site_dir(name: str) -> Path:
    return Path(os.environ.get(f"XSTACK_TORCH_SITE_{name.upper()}", str(EXTRA_SITES[name])))


def _require_site(name: str, probe: str, front: bool = False) -> None:
    """Put the dependency site on sys.path (after the venv, so its torch/numpy win; ``front``
    for a site whose pins must shadow the venv's copy, e.g. transformers 4.x needs
    huggingface_hub < 1.0) and fail loudly, with the install hint, if the probe package is
    not there."""
    d = site_dir(name)
    if not (d / probe).exists() and not (d / f"{probe}.py").exists():
        raise ImportError(
            f"{probe} not found under {d} (site {name!r}); install with: uv pip install "
            f"--python <venv>/bin/python --target {d} --no-deps <packages>"
        )
    if str(d) in sys.path:
        sys.path.remove(str(d))
    if front:
        sys.path.insert(0, str(d))
    else:
        sys.path.append(str(d))


Transform = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
SelfTests = Callable[[dict[str, torch.Tensor]], list[tuple[str, Transform, str]]]


@dataclass(frozen=True)
class RepoEntry:
    repo: str
    source_dir: str  # relative to REPO_ROOT
    config_note: str  # canonical graded config, human-readable, goes into the report
    build: Callable[[], torch.nn.Module] | None = None
    build_from_npz: Callable[[Any], torch.nn.Module] | None = None
    exempt_buffers: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()
    tie_groups: tuple[tuple[str, ...], ...] = ()
    tied_or_untied_ok: bool = False
    alt_shapes: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    pad_value: float = 1.0
    extra_strip_prefixes: tuple[str, ...] = ()
    extra_self_tests: SelfTests | None = None
    not_applicable: str | None = None  # one sentence: why stage 2 has no meaning here
    stub_reason: str | None = None  # set when the constructor cannot be built on CPU

    @property
    def source_path(self) -> Path:
        return REPO_ROOT / self.source_dir

    def self_test_cases(self, sd: dict[str, torch.Tensor]) -> list[tuple[str, Transform, str]]:
        return self.extra_self_tests(sd) if self.extra_self_tests else []


# --------------------------------------------------------------------------
# import helpers
# --------------------------------------------------------------------------

# top-level names several vendored repos use for their own packages; a stale entry in
# sys.modules from one repo must not satisfy another repo's import in the same process
GENERIC_TOP_LEVEL = (
    "models", "model", "utils", "util", "modules", "layers", "src", "common", "lib", "data",
    "datasets", "configs", "qm9", "training", "nanochat", "x_transformers",
)


def _purge_modules(names: Sequence[str]) -> None:
    for k in list(sys.modules):
        top = k.split(".")[0]
        if top in names:
            del sys.modules[k]


def _assert_vendored(module: types.ModuleType, root: Path) -> None:
    """The whole point is the UNMODIFIED source model: refuse a shadowing install."""
    file = getattr(module, "__file__", None)
    # a namespace package (no __init__.py, e.g. generative_recommenders) has no __file__
    locations = [file] if file else list(getattr(module, "__path__", []) or [])
    if not locations or not all(
        Path(loc).resolve().is_relative_to(root.resolve()) for loc in locations
    ):
        raise ImportError(
            f"{module.__name__} resolved to {locations}, not the vendored tree {root}"
        )


def _import_package_module(
    root: Path, dotted: str, purge: Sequence[str] = ()
) -> types.ModuleType:
    """Import ``dotted`` (a package-qualified module) with ``root`` first on sys.path."""
    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)
    _purge_modules(purge)
    top = dotted.split(".")[0]
    mod = importlib.import_module(dotted)
    _assert_vendored(sys.modules[top], root)
    return mod


def _import_package_skipping_init(
    root: Path, pkg: str, submodules: tuple[str, ...]
) -> types.ModuleType:
    """Register ``pkg`` as a package rooted at ``root/pkg`` WITHOUT executing its
    ``__init__.py`` (which may pull in heavy optional dependencies), then
    import the named submodules by their real files. Returns the last one."""
    pkg_dir = root / pkg
    _purge_modules([pkg])
    stub = types.ModuleType(pkg)
    stub.__path__ = [str(pkg_dir)]  # makes ``pkg.x`` resolve to root/pkg/x.py
    stub.__file__ = str(pkg_dir / "__init__.py")
    sys.modules[pkg] = stub
    _assert_vendored(sys.modules[pkg], root)
    mod = None
    for sub in submodules:
        mod = importlib.import_module(f"{pkg}.{sub}")
        _assert_vendored(mod, root)
    assert mod is not None
    return mod


def _import_file(
    path: Path, unique_name: str, extra_sys_path: tuple[Path, ...] = ()
) -> types.ModuleType:
    """Import a single file under a unique module name (several repos ship a ``model.py``)."""
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    for p in extra_sys_path:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _class_block(path: Path, class_name: str) -> str:
    """The source text of ``class <class_name>`` up to the next top-level statement.
    For scripts whose module body parses argv, builds data loaders or trains."""
    lines = path.read_text().splitlines(keepends=True)
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith(f"class {class_name}")), None
    )
    if start is None:
        raise RuntimeError(f"{path}: class {class_name} not found (pin changed?)")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln[:1] not in (" ", "\t", "\n", "#", "") and not ln.startswith(")"):
            end = j
            break
    return "".join(lines[start:end])


def _exec_class(path: Path, class_name: str, namespace: dict[str, Any]) -> type:
    """Exec one class definition from a script (byte-identical text) in ``namespace``."""
    code = compile(_class_block(path, class_name), str(path), "exec")
    exec(code, namespace)  # noqa: S102 - vendored, pinned
    return namespace[class_name]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def _cpu_only():
    """Some sources call ``.cuda()`` inside the constructor; make it a no-op on CPU."""
    orig_module, orig_tensor = torch.nn.Module.cuda, torch.Tensor.cuda
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.nn.Module.cuda, torch.Tensor.cuda = orig_module, orig_tensor


_TORCH_NS = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}


# --------------------------------------------------------------------------
# per-repo builders (all on CPU, deterministic shapes; values are irrelevant)
# --------------------------------------------------------------------------

_MINGPT_ROOT = "tasks/mingpt/source_torch/minGPT"


def _build_mingpt() -> torch.nn.Module:
    # projects/adder/adder.py: model_type gpt-nano (n_layer 3, n_head 3, n_embd 48),
    # ndigit 2 -> vocab_size 10, block_size 3*ndigit + 1 - 1 = 6.
    model_mod = _import_package_module(REPO_ROOT / _MINGPT_ROOT, "mingpt.model")
    cfg = model_mod.GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 10
    cfg.block_size = 6
    with torch.device("cpu"), _quiet():
        return model_mod.GPT(cfg)


# The source ships two demo models and the blind prompt names neither;
# five ports exported the chargpt model (gpt-mini, vocab 65, block 128) and failed a load check built
# for the adder (gpt-nano, vocab 10, block 6) although each export is name- and shape-exact. The
# geometry is now read from the export and accepted only if it is one of the two demo geometries.
# Entries changed after reports were written: stage2_rerun.py re-decides a turn whose report predates the stamp.

_MINGPT_DEMOS = {(3, 48, 10, 6): ("gpt-nano", "adder demo"), (6, 192, 65, 128): ("gpt-mini", "chargpt demo")}


def _build_mingpt_from_npz(npz=None) -> torch.nn.Module:
    from check_weight_load import normalise_key  # noqa: PLC0415 - sibling script

    if npz is None:
        return _build_mingpt()
    keys = {normalise_key(k): k for k in npz.keys()}
    layers = {int(m.group(1)) for k in keys for m in [re.match(r"transformer\.h\.(\d+)\.", k)] if m}
    geom = None
    try:
        geom = (max(layers) + 1, npz.shape(keys["transformer.ln_f.weight"])[0],
                npz.shape(keys["transformer.wte.weight"])[0], npz.shape(keys["transformer.wpe.weight"])[0])
    except (KeyError, ValueError, IndexError):
        pass
    if geom not in _MINGPT_DEMOS:
        model = _build_mingpt()
        model._stage2_note = f"export geometry {geom} is neither demo (adder 3/48/10/6, chargpt 6/192/65/128); built the adder model"
        return model
    model_type, name = _MINGPT_DEMOS[geom]
    model_mod = _import_package_module(REPO_ROOT / _MINGPT_ROOT, "mingpt.model")
    cfg = model_mod.GPT.get_default_config()
    cfg.model_type, cfg.vocab_size, cfg.block_size = model_type, geom[2], geom[3]
    with torch.device("cpu"), _quiet():
        model = model_mod.GPT(cfg)
    model._stage2_note = f"export geometry {geom}: {name} ({model_type}, vocab {geom[2]}, block {geom[3]})"
    return model


_NANOGPT_ROOT = "tasks/nanogpt/source_torch/nanogpt"


def _build_nanogpt() -> torch.nn.Module:
    # config/train_gpt2.py (GPT-2 124M reproduction): n_layer 12, n_head 12, n_embd 768,
    # vocab 50304 (padded), block 1024, bias False, dropout 0.0.
    root = REPO_ROOT / _NANOGPT_ROOT
    model_mod = _import_file(root / "model.py", "torch2jax_stage2_nanogpt_model")
    _assert_vendored(model_mod, root)
    cfg = model_mod.GPTConfig(
        n_layer=12, n_head=12, n_embd=768, vocab_size=50304, block_size=1024,
        bias=False, dropout=0.0,
    )
    with torch.device("cpu"), _quiet():
        return model_mod.GPT(cfg)


_LORA_ROOT = "tasks/lora/source_torch/lora"


def _build_lora() -> torch.nn.Module:
    # examples/NLG/src/gpt2_ft.py, model_card gpt2.md with the README E2E recipe:
    # GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=4, lora_attn_alpha=32,
    # lora_dropout=0.1); the LoRA lives on the fused c_attn (loralib.MergedLinear,
    # enable_lora=[True, False, True], fan_in_fan_out=True).
    root = REPO_ROOT / _LORA_ROOT
    src = root / "examples" / "NLG" / "src"
    _purge_modules(["loralib"])
    model_mod = _import_file(
        src / "model.py", "torch2jax_stage2_lora_model", extra_sys_path=(root,)
    )
    _assert_vendored(model_mod, root)
    _assert_vendored(sys.modules["loralib"], root)
    cfg = model_mod.GPT2Config(
        n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=4, lora_attn_alpha=32, lora_dropout=0.1,
    )
    with torch.device("cpu"):
        model = model_mod.GPT2LMModel(cfg)
    model.set_tied()  # the source re-ties after load_weight (model.py:449); a no-op here
    return model


_MODDED_ROOT = "tasks/modded_nanogpt/source_torch/modded-nanogpt"
_MODDED_MODNAME = "torch2jax_stage2_modded_nanogpt_train_gpt"


class _DistStub:
    """Stands in for ``torch.distributed`` inside the exec'd script: the GPT constructor
    calls ``dist.get_world_size()`` to size the ``scalars`` pad (train_gpt.py:407)."""

    def __init__(self, world_size: int):
        self._world_size = world_size

    def get_world_size(self) -> int:
        return self._world_size

    def __getattr__(self, name):  # everything else falls through to the real module
        return getattr(torch.distributed, name)


def _load_modded_definitions(world_size: int = 8) -> types.ModuleType:
    """Exec only the class-definition prefix of train_gpt.py (a script whose module body
    opens CUDA, joins a process group and trains). Cut before ``class Hyperparameters``;
    neutralise the two CUDA/argv lines above the definitions. The text that is exec'd is
    otherwise byte-identical to the pinned file."""
    if _MODDED_MODNAME in sys.modules:  # custom_op registration must happen once per process
        return sys.modules[_MODDED_MODNAME]
    path = REPO_ROOT / _MODDED_ROOT / "train_gpt.py"
    text = path.read_text()
    marker = "@dataclass\nclass Hyperparameters:"
    cut = text.find(marker)
    if cut < 0:
        raise RuntimeError(f"{path}: Hyperparameters marker not found; pin changed?")
    prefix = text[:cut]
    edits = {
        (
            "with open(sys.argv[0]) as f:\n"
            "    code = f.read() # read the code of this file ASAP, for logging\n"
        ): "code = ''\n",
        (
            'torch.empty(1, device="cuda", requires_grad=True).backward()'
            " # prevents a bug on some systems\n"
        ): "",
    }
    for old, new in edits.items():
        if old not in prefix:
            raise RuntimeError(f"{path}: expected line not found (pin changed?): {old.strip()!r}")
        prefix = prefix.replace(old, new, 1)
    mod = types.ModuleType(_MODDED_MODNAME)
    mod.__file__ = str(path)
    sys.modules[_MODDED_MODNAME] = mod
    try:
        exec(compile(prefix, str(path), "exec"), mod.__dict__)  # noqa: S102 - vendored, pinned
    except Exception:
        del sys.modules[_MODDED_MODNAME]
        raise
    mod.dist = _DistStub(world_size)
    return mod


def _build_modded_nanogpt() -> torch.nn.Module:
    # train_gpt.py:608-611 at world_size 8 (the canonical 8xH100 run):
    # GPT(vocab_size=50257 -> padded 50304, num_layers=12, num_heads=6, model_dim=768,
    # max_seq_len=max(train_seq_len 49152, val_seq_len 262144)), embeddings cast bf16.
    mod = _load_modded_definitions(world_size=8)
    with torch.device("cpu"):
        model = mod.GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768,
                        max_seq_len=262144)
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    return model


_GENREC_ROOT = "tasks/generative_recommenders/source_torch/generative-recommenders"


def _build_generative_recommenders() -> torch.nn.Module:
    # configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin bound onto train_fn
    # (trainer/train.py:154-199) and hstu_encoder (encoder_utils.py:70-111). Built here
    # by calling the same constructors with the same values, bypassing gin (not in the
    # venv) and the fbgemm ops (forward-only). ml-20m expected_max_item_id=131262
    # (data/preprocessor.py:451) -> item table 131,263 rows; gr_output_length 10 (train.py:116).
    root = REPO_ROOT / _GENREC_ROOT
    pkg = "generative_recommenders.modeling.sequential"
    emb = _import_package_module(root, f"{pkg}.embedding_modules")
    pre = _import_package_module(root, f"{pkg}.input_features_preprocessors")
    post = _import_package_module(root, f"{pkg}.output_postprocessors")
    hstu = _import_package_module(root, f"{pkg}.hstu")
    dot = _import_package_module(
        root, "generative_recommenders.rails.similarities.dot_product_similarity_fn"
    )
    max_sequence_length, gr_output_length, d = 200, 10, 256
    with torch.device("cpu"), _quiet():
        embedding_module = emb.LocalEmbeddingModule(num_items=131262, item_embedding_dim=d)
        input_preproc = pre.LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=max_sequence_length + gr_output_length + 1,
            embedding_dim=d,
            dropout_rate=0.2,
        )
        output_postproc = post.L2NormEmbeddingPostprocessor(embedding_dim=d, eps=1e-6)
        return hstu.HSTU(
            embedding_module=embedding_module,
            similarity_module=dot.DotProductSimilarity(),
            input_features_preproc_module=input_preproc,
            output_postproc_module=output_postproc,
            max_sequence_len=max_sequence_length,
            max_output_len=gr_output_length + 1,
            embedding_dim=d,
            num_blocks=16,
            num_heads=8,
            attention_dim=32,  # dqk
            linear_dim=32,  # dv
            linear_dropout_rate=0.2,
            attn_dropout_rate=0.0,
            linear_config="uvqk",
            linear_activation="silu",
            normalization="rel_bias",
            concat_ua=False,
            enable_relative_attention_bias=True,
            verbose=False,
        )


_LLAMA2C_ROOT = "tasks/llama2c/source_torch/llama2c"


def _build_llama2c() -> torch.nn.Module:
    # train.py defaults (the stories15M recipe): dim 288, 6 layers, 6 heads, 6 kv heads,
    # vocab 32000, multiple_of 32, max_seq_len 256, dropout 0.0.
    root = REPO_ROOT / _LLAMA2C_ROOT
    model_mod = _import_file(root / "model.py", "torch2jax_stage2_llama2c_model")
    _assert_vendored(model_mod, root)
    args = model_mod.ModelArgs(
        dim=288, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, multiple_of=32,
        max_seq_len=256, dropout=0.0,
    )
    with torch.device("cpu"):
        return model_mod.Transformer(args)


_AST_ROOT = "tasks/ast/source_torch/ast"


def _build_ast() -> torch.nn.Module:
    # egs/speechcommands/run_sc.sh -> traintest.py: ASTModel(label_dim=35, fstride=10,
    # tstride=10, input_fdim=128, input_tdim=128, imagenet_pretrain=True,
    # audioset_pretrain=False, model_size='base384'). Constructed here with
    # imagenet_pretrain=False so no checkpoint download happens; the state_dict names and
    # shapes are identical (the pretrained branch only fills values). The source hard-asserts
    # timm==0.4.5 (ast_models.py:50) and imports wget at module level.
    _require_site("timm045", "timm")
    root = REPO_ROOT / _AST_ROOT
    mod = _import_file(
        root / "src" / "models" / "ast_models.py", "torch2jax_stage2_ast_models"
    )
    _assert_vendored(mod, root)
    with torch.device("cpu"):
        return mod.ASTModel(
            label_dim=35, fstride=10, tstride=10, input_fdim=128, input_tdim=128,
            imagenet_pretrain=False, audioset_pretrain=False, model_size="base384",
            verbose=False,
        )


_AIRBENCH_ROOT = "tasks/airbench/source_torch/cifar10-airbench"


def _build_airbench() -> torch.nn.Module:
    # legacy/airbench94.py:make_net() (the graded script; hyp['net'] widths 64/256/256,
    # batchnorm_momentum 0.6, scaling_factor 1/9). The function ends with .half().cuda();
    # .cuda is made a no-op for the call. The whitening conv 0.weight is a frozen parameter
    # (requires_grad False) that the port must still export; BN counters are exempt for every task.
    _require_site("timm045", "torchvision")  # the script imports torchvision at module level
    root = REPO_ROOT / _AIRBENCH_ROOT
    mod = _import_file(root / "legacy" / "airbench94.py", "torch2jax_stage2_airbench94")
    _assert_vendored(mod, root)
    with torch.device("cpu"), _cpu_only():
        return mod.make_net()


_MAKEMORE_ROOT = "tasks/makemore/source_torch/makemore"


def _build_makemore() -> torch.nn.Module:
    # makemore.py --type transformer defaults on names.txt: vocab 26 letters + 1 = 27,
    # block_size = max word length 15 + 1 = 16, n_layer 4, n_head 4, n_embd 64, n_embd2 64.
    _require_site("tensorboard", "tensorboard")  # makemore.py:28 imports torch.utils.tensorboard
    root = REPO_ROOT / _MAKEMORE_ROOT
    mod = _import_file(root / "makemore.py", "torch2jax_stage2_makemore")
    _assert_vendored(mod, root)
    cfg = mod.ModelConfig(vocab_size=27, block_size=16, n_layer=4, n_head=4, n_embd=64, n_embd2=64)
    with torch.device("cpu"), _quiet():
        return mod.Transformer(cfg)


_NANOCHAT_ROOT = "tasks/nanochat_core/source_torch/nanochat_repo"


def _build_nanochat_core() -> torch.nn.Module:
    # runs/speedrun.sh: base_train --depth=24 -> build_model_meta: model_dim = 24*64 = 1536,
    # head_dim 128 -> n_head = n_kv_head = 12 (MHA at this pin), sequence_len 2048, vocab
    # 32768 (already a multiple of 64), window_pattern SSSL. Built on CPU (not meta) so
    # the tensors exist; init_weights() is not needed for names/shapes.
    root = REPO_ROOT / _NANOCHAT_ROOT
    gpt = _import_package_module(root, "nanochat.gpt", purge=("nanochat",))
    cfg = gpt.GPTConfig(
        sequence_len=2048, vocab_size=32768, n_layer=24, n_head=12, n_kv_head=12, n_embd=1536,
        window_pattern="SSSL",
    )
    # 1.38B parameters = 5.5 GB in fp32; a login node's watchdog may kill processes near 4 GB.
    # XSTACK_TORCH_LOWMEM=1 constructs in bf16 (the source's own compute dtype); names and
    # shapes are unchanged and dtype is not compared.
    dtype = torch.bfloat16 if os.environ.get("XSTACK_TORCH_LOWMEM") else torch.float32
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cpu"), _quiet():
            return gpt.GPT(cfg)
    finally:
        torch.set_default_dtype(prev)


_NANOVLM_ROOT = "tasks/nanovlm/source_torch/nanoVLM"


def _build_nanovlm() -> torch.nn.Module:
    # train.py: VisionLanguageModel(VLMConfig(), load_backbone=True) downloads SigLIP-B/16
    # and SmolLM2-135M; load_backbone=False builds the same modules from the config with no
    # download (identical names/shapes). lm_tie_weights True -> head tied to token_embedding.
    root = REPO_ROOT / _NANOVLM_ROOT
    vlm = _import_package_module(root, "models.vision_language_model", purge=("models",))
    cfg_mod = _import_package_module(root, "models.config")
    with torch.device("cpu"), _quiet():
        return vlm.VisionLanguageModel(cfg_mod.VLMConfig(), load_backbone=False)


_PATCHTST_ROOT = "tasks/patchtst/source_torch/PatchTST/PatchTST_supervised"


def _build_patchtst() -> torch.nn.Module:
    # scripts/PatchTST/traffic.sh, pred_len 96, with run_longExp.py defaults for the rest.
    root = REPO_ROOT / _PATCHTST_ROOT
    mod = _import_package_module(root, "models.PatchTST", purge=("models", "layers"))
    configs = SimpleNamespace(
        enc_in=862, seq_len=336, pred_len=96, e_layers=3, n_heads=16, d_model=128, d_ff=256,
        dropout=0.2, fc_dropout=0.2, head_dropout=0.0, patch_len=16, stride=8,
        padding_patch="end", revin=1, affine=0, subtract_last=0, decomposition=0,
        kernel_size=25, individual=0,
    )
    with torch.device("cpu"):
        return mod.Model(configs)


_S4_ROOT = "tasks/s4/source_torch/s4"


def _build_s4() -> torch.nn.Module:
    # example.py defaults (CIFAR-10, RGB): S4Model(d_input=3, d_output=10, d_model=128,
    # n_layers=4, dropout=0.1, prenorm=False); each layer S4D(d_model, dropout,
    # transposed=True, lr=min(0.001, args.lr=0.01)). The script trains at module level,
    # so the class definition is exec'd with the S4D import and ``args`` supplied.
    root = REPO_ROOT / _S4_ROOT
    s4d = _import_package_module(root, "models.s4.s4d", purge=("models", "src"))
    ns = dict(_TORCH_NS)
    # example.py:38-46 picks dropout_fn by torch version: nn.Dropout1d for torch >= 1.12
    ns.update(S4D=s4d.S4D, args=SimpleNamespace(lr=0.01), dropout_fn=torch.nn.Dropout1d)
    s4model = _exec_class(root / "example.py", "S4Model", ns)
    with torch.device("cpu"):
        return s4model(d_input=3, d_output=10, d_model=128, n_layers=4, dropout=0.1, prenorm=False)


_TENSORF_ROOT = "tasks/tensorf/source_torch/TensoRF"
_TENSORF_AABB = [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]  # dataLoader/blender.py:23
_TENSORF_INIT_GRID = [128, 128, 128]  # N_to_reso(N_voxel_init 2097156, aabb) before shrink


def _tensorf_grid_from_export(npz) -> tuple[list[int], list[list[float]], str]:
    """gridSize/aabb the way the source's load() takes them from ckpt['kwargs']: from the
    export's ``gridSize``/``aabb`` metadata arrays when present, else gridSize inferred from
    the three density lines (line i has shape (1, c, gridSize[vecMode[i]], 1), vecMode
    [2,1,0]), which the plane shapes are then checked against."""
    aabb = _TENSORF_AABB
    if npz is None:
        return list(_TENSORF_INIT_GRID), aabb, "no export: initial 128^3 grid"
    if "aabb" in npz:
        aabb = np.asarray(npz.array("aabb"), dtype=np.float64).reshape(2, 3).tolist()
    if "gridSize" in npz:
        gs = [int(x) for x in np.asarray(npz.array("gridSize")).reshape(-1)]
        how = "gridSize from export metadata"
    else:
        lines = [f"density_line.{i}" for i in range(3)]
        if not all(k in npz for k in lines):
            return list(_TENSORF_INIT_GRID), aabb, "no gridSize/density_line in export: 128^3"
        # line i -> gridSize[vecMode[i]] with vecMode = [2, 1, 0]
        per_line = [npz.shape(k)[2] for k in lines]
        gs = [per_line[2], per_line[1], per_line[0]]
        how = "gridSize inferred from density_line shapes"
    if len(gs) != 3 or min(gs) < 1:
        raise ValueError(f"tensorf: bad gridSize {gs}")
    return gs, aabb, how


def _build_tensorf(npz=None) -> torch.nn.Module:
    # configs/lego.txt on train.py:135: TensorVMSplit(aabb, reso, device, density_n_comp=
    # [16,16,16], appearance_n_comp=[48,48,48], app_dim=27, near_far=[2,6], shadingMode
    # MLP_Fea, alphaMask_thres 1e-4, density_shift -10, distance_scale 25, pos_pe 6,
    # view_pe 2, fea_pe 2, featureC 128, step_ratio 0.5, fea2denseAct softplus). The final
    # gridSize depends on the alpha-mask shrink at iteration 2000, so it is taken from the
    # export (see _tensorf_grid_from_export); the 12 plane/line shapes are then a
    # consistency check against that one grid.
    root = REPO_ROOT / _TENSORF_ROOT
    trf = _import_package_module(root, "models.tensoRF", purge=("models", "utils"))
    grid, aabb, how = _tensorf_grid_from_export(npz)
    with torch.device("cpu"):
        model = trf.TensorVMSplit(
            torch.tensor(aabb, dtype=torch.float32), grid, torch.device("cpu"),
            density_n_comp=[16, 16, 16], appearance_n_comp=[48, 48, 48], app_dim=27,
            near_far=[2.0, 6.0], shadingMode="MLP_Fea", alphaMask_thres=1e-4,
            density_shift=-10, distance_scale=25, pos_pe=6, view_pe=2, fea_pe=2,
            featureC=128, step_ratio=0.5, fea2denseAct="softplus",
        )
    model._stage2_note = f"{how}: gridSize {grid}, aabb {aabb}"
    return model


def _tensorf_self_tests(sd):
    """Export-driven construction: a resized grid must PASS (all 12 planes/lines consistent),
    an inconsistent plane must FAIL, and the metadata arrays are tolerated extras."""
    def resize(grid):
        def t(a):
            b = dict(a)
            vec, mat = [2, 1, 0], [[0, 1], [0, 2], [1, 2]]
            for kind in ("density", "app"):
                for i in range(3):
                    c = b[f"{kind}_plane.{i}"].shape[1]
                    shp = (1, c, grid[mat[i][1]], grid[mat[i][0]])
                    b[f"{kind}_plane.{i}"] = np.zeros(shp, np.float32)
                    b[f"{kind}_line.{i}"] = np.zeros((1, c, grid[vec[i]], 1), np.float32)
            return b
        return t

    def with_meta(a):
        b = resize([300, 296, 288])(a)
        b["gridSize"] = np.array([300, 296, 288], np.int64)
        b["aabb"] = np.array(_TENSORF_AABB, np.float32)
        b["alphaMask.shape"] = np.array([10, 10, 10], np.int64)
        b["alphaMask.mask"] = np.zeros((125,), np.uint8)
        b["alphaMask.aabb"] = np.array(_TENSORF_AABB, np.float32)
        return b

    def inconsistent(a):
        b = resize([300, 296, 288])(a)
        b["app_plane.1"] = b["app_plane.1"][:, :, :-1, :]  # gz-1 rows: disagrees with the lines
        return b

    return [
        ("grid_resized_300x296x288", resize([300, 296, 288]), "PASS"),
        ("grid_from_metadata_plus_alphamask", with_meta, "PASS"),
        ("grid_inconsistent_plane", inconsistent, "FAIL"),
    ]


_TGN_ROOT = "tasks/tgn/source_torch/tgn"


def _build_tgn() -> torch.nn.Module:
    # README: train_self_supervised.py --use_memory --prefix tgn-attn (wikipedia). Defaults:
    # n_layer 1, n_head 2, drop_out 0.1, message_dim 100, memory_dim 172, graph_attention
    # embeddings, identity messages, last aggregator, gru updater, n_degree 10. Wikipedia:
    # 9,228 nodes x 172 node features, 157,475 edges x 172 edge features (zeros here; only
    # the shapes matter). The neighbor finder is not used at construction.
    root = REPO_ROOT / _TGN_ROOT
    tgn = _import_package_module(root, "model.tgn", purge=("model", "modules", "utils"))
    node_features = np.zeros((9228, 172), dtype=np.float32)
    edge_features = np.zeros((157475, 172), dtype=np.float32)
    with torch.device("cpu"):
        return tgn.TGN(
            neighbor_finder=None, node_features=node_features, edge_features=edge_features,
            device=torch.device("cpu"), n_layers=1, n_heads=2, dropout=0.1, use_memory=True,
            message_dimension=100, memory_dimension=172, embedding_module_type="graph_attention",
            message_function="identity", aggregator_type="last", memory_updater_type="gru",
            n_neighbors=10,
        )


_XTR_ROOT = "tasks/x_transformers/source_torch/x_transformers_repo"


def _build_x_transformers() -> torch.nn.Module:
    # train_enwik8.py:53-78: TransformerWrapper(num_tokens=256, max_seq_len=1024,
    # attn_layers=Decoder(dim=512, depth=6, heads=8, rotary_pos_emb=False,
    # polar_pos_emb=True, pre_and_post_norm=True)), then AutoregressiveWrapper (which adds
    # the ``net.`` prefix; stripped by the entry). The venv has a pip x_transformers: the
    # vendored tree is forced first and its provenance asserted.
    root = REPO_ROOT / _XTR_ROOT
    xt = _import_package_module(root, "x_transformers", purge=("x_transformers",))
    with torch.device("cpu"), _quiet():
        return xt.TransformerWrapper(
            num_tokens=256, max_seq_len=1024,
            attn_layers=xt.Decoder(dim=512, depth=6, heads=8, rotary_pos_emb=False,
                                   polar_pos_emb=True, pre_and_post_norm=True),
        )


# ---- entries needing site dependencies beyond timm/torchvision ----------------


_DEEPSPEECH_ROOT = "tasks/deepspeech/source_torch/deepspeech.pytorch"


def _build_deepspeech() -> torch.nn.Module:
    # train.py -> DeepSpeech(labels=labels.json (29), model_cfg=BiDirectionalConfig()
    # (lstm, hidden 1024, 5 layers), precision=32, optim_cfg=AdamConfig(),
    # spect_cfg=SpectConfig()). A LightningModule: needs pytorch_lightning + omegaconf
    # (+ torchmetrics, Levenshtein, six for the validation/decoder imports).
    _require_site("lightning", "pytorch_lightning")
    root = REPO_ROOT / _DEEPSPEECH_ROOT
    # configs/train_config.py (written for Python 3.8) uses dataclass instances as field
    # defaults; Python >= 3.11 rejects unhashable defaults at class creation. For the
    # config import only, the dataclass decorator is wrapped with unsafe_hash=True so the
    # source's own default objects are accepted unchanged.
    import dataclasses as _dc  # noqa: PLC0415

    orig_dataclass = _dc.dataclass

    def hashable_dataclass(cls=None, /, **kw):
        def wrap(c):
            if c.__module__.startswith("deepspeech_pytorch."):  # the source's classes only
                kw.setdefault("unsafe_hash", True)
            return orig_dataclass(c, **kw)

        return wrap if cls is None else wrap(cls)

    _dc.dataclass = hashable_dataclass
    try:  # configs first (omegaconf only); the model import pulls in pytorch_lightning
        cfgs = _import_package_module(
            root, "deepspeech_pytorch.configs.train_config", purge=("deepspeech_pytorch",)
        )
    finally:
        _dc.dataclass = orig_dataclass
    model_mod = importlib.import_module("deepspeech_pytorch.model")
    _assert_vendored(model_mod, root)
    labels = json.loads((root / "labels.json").read_text())
    with torch.device("cpu"):
        return model_mod.DeepSpeech(
            labels=labels, model_cfg=cfgs.BiDirectionalConfig(), precision=32,
            optim_cfg=cfgs.AdamConfig(), spect_cfg=cfgs.SpectConfig(),
        )


_SIMCSE_ROOT = "tasks/simcse/source_torch/simcse"


def _build_simcse() -> torch.nn.Module:
    # run_unsup_example.sh: bert-base-uncased, --pooler_type cls, --temp 0.05,
    # --mlp_only_train, no MLM. BertForCL(BertConfig() = bert-base-uncased values,
    # model_args=...); add_pooling_layer=False so no bert.pooler.*; head is mlp.dense.
    _require_site("transformers", "transformers", front=True)
    root = REPO_ROOT / _SIMCSE_ROOT
    # simcse/__init__.py imports tool.py (sklearn, for the inference helper); skip it
    models = _import_package_skipping_init(root, "simcse", ("models",))
    from transformers import BertConfig  # noqa: PLC0415 - dependency site, resolved above

    model_args = SimpleNamespace(
        pooler_type="cls", temp=0.05, hard_negative_weight=0.0, do_mlm=False, mlm_weight=0.1,
        mlp_only_train=True,
    )
    with torch.device("cpu"), _quiet():
        return models.BertForCL(BertConfig(), model_args=model_args)


_TABR_ROOT = "tasks/tabr/source_torch/tabular-dl-tabr"


def _build_tabr() -> torch.nn.Module:
    # exp/tabr/covtype/2-lr-evaluation/0.toml [model] on covtype (10 numerical + 44 binary
    # features, no categoricals, 7 classes). bin/tabr.py imports faiss, delu, loguru,
    # tensorboard and the repo's lib (scipy, sklearn, tomli).
    _require_site("scientific", "scipy")
    _require_site("tabr", "faiss")
    _require_site("tensorboard", "tensorboard")
    root = REPO_ROOT / _TABR_ROOT
    # lib/env.py reads $PROJECT_DIR at import (the source's own convention: the repo root)
    os.environ.setdefault("PROJECT_DIR", str(root))
    lib = _import_package_module(root, "lib", purge=("lib", "bin"))
    _assert_vendored(lib, root)
    tabr = _import_file(root / "bin" / "tabr.py", "torch2jax_stage2_tabr_bin")
    with torch.device("cpu"):
        return tabr.Model(
            n_num_features=10, n_bin_features=44, cat_cardinalities=[], n_classes=7,
            num_embeddings={"type": "LREmbeddings", "d_embedding": 36}, d_main=130,
            d_multiplier=2.0, encoder_n_blocks=1, predictor_n_blocks=1,
            mixer_normalization="auto", context_dropout=0.0437357558538884,
            dropout0=0.5343904237887059, dropout1=0.0, normalization="LayerNorm",
            activation="ReLU",
        )


_TDMPC2_ROOT = "tasks/tdmpc2/source_torch/tdmpc2/tdmpc2"


class _Cfg(SimpleNamespace):
    """config.yaml values as an attribute bag with the .get() the source's helpers use."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_tdmpc2() -> torch.nn.Module:
    # config.yaml defaults (5M single-task: latent 512, mlp 512, enc_dim 256, 2 enc layers,
    # num_q 5, num_bins 101, simnorm 8, dropout 0.01) with walker-run dims (obs state 24,
    # action 6; single task -> task_dim 0, no task embedding; episodic false). parse_cfg
    # derivations reproduced: bin_size = (vmax-vmin)/(num_bins-1), multitask False,
    # tasks [task]. WorldModel needs tensordict.
    _require_site("tensordict", "tensordict")
    root = REPO_ROOT / _TDMPC2_ROOT
    wm = _import_package_module(root, "common.world_model", purge=("common",))
    cfg = _Cfg(
        task="walker-run", obs="state", episodic=False, multitask=False, tasks=["walker-run"],
        task_dim=0, obs_shape={"state": (24,)}, action_dim=6, action_dims=None,
        latent_dim=512, mlp_dim=512, enc_dim=256, num_enc_layers=2, num_channels=32,
        num_q=5, num_bins=101, vmin=-10, vmax=10, bin_size=20 / 100, simnorm_dim=8,
        dropout=0.01, log_std_min=-10, log_std_max=2, model_size=5,
    )
    with torch.device("cpu"), _quiet():
        return wm.WorldModel(cfg)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

_CAUSAL_MASK = r"transformer\.h\.\d+\.attn\.bias"

REGISTRY: dict[str, RepoEntry] = {
    # ---- the five original entries ---------------------------------------------------
    "mingpt": RepoEntry(
        repo="mingpt", source_dir=_MINGPT_ROOT,
        config_note="one of the source's two demo models, read from the export: adder (gpt-nano: n_layer 3, "
        "n_head 3, n_embd 48; vocab 10, block 6) or chargpt (gpt-mini: 6 layers, 192 wide; vocab 65, block 128)",
        build=_build_mingpt,
        build_from_npz=_build_mingpt_from_npz,
        exempt_buffers=(_CAUSAL_MASK,),  # causal mask, (1,1,6,6) or (1,1,128,128)
    ),
    "nanogpt": RepoEntry(
        repo="nanogpt", source_dir=_NANOGPT_ROOT,
        config_note="GPT-2 124M reproduction: GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768, "
        "vocab_size=50304, block_size=1024, bias=False, dropout=0.0))",
        build=_build_nanogpt,
        # only registered when F.scaled_dot_product_attention is absent (model.py:45-49);
        # a port may export it, the torch-2.13 source does not have it.
        exempt_buffers=(_CAUSAL_MASK,),
        tie_groups=(("transformer.wte.weight", "lm_head.weight"),),  # model.py:138
    ),
    "lora": RepoEntry(
        repo="lora", source_dir=_LORA_ROOT,
        config_note="GPT2LMModel(GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=4, "
        "lora_attn_alpha=32, lora_dropout=0.1)) (GPT-2 medium + LoRA on fused c_attn)",
        build=_build_lora,
        exempt_buffers=(_CAUSAL_MASK,),  # causal mask (1,1,1024,1024)
        tie_groups=(("transformer.wte.weight", "lm_head.decoder.weight"),),  # model.py:287-288
    ),
    "modded_nanogpt": RepoEntry(
        repo="modded_nanogpt", source_dir=_MODDED_ROOT,
        config_note="GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, "
        "max_seq_len=262144) at world_size 8, embeddings bf16; blocks.7 has no attention; "
        "scalars is one packed vector of 64 (ws 8) or 60 (ws 4)",
        build=_build_modded_nanogpt,
        # Rotary cos/sin are non-persistent buffers (nn.Buffer(persistent=False)) and are
        # not in the state_dict; nothing to exempt. Untied head (lm_head.weight separate).
        alt_shapes={"scalars": ((60,),)},
        pad_value=1.0,  # the source pads scalars with ones (train_gpt.py:412)
    ),
    # ---- second wave --------------------------------------------------------------------
    "generative_recommenders": RepoEntry(
        repo="generative_recommenders", source_dir=_GENREC_ROOT,
        config_note="ML-20M HSTU-large (hstu-sampled-softmax-n128-large-final.gin): 16 blocks, "
        "8 heads, dqk=dv=32, d=256, item table 131263x256, pos_emb 211x256, fused _uvqk, "
        "rel time+pos bias; no LayerNorm params, no ties",
        build=_build_generative_recommenders,
        exempt_buffers=(r"_attn_mask",),  # triu(ones(211,211), 1), deterministic
    ),
    "llama2c": RepoEntry(
        repo="llama2c", source_dir=_LLAMA2C_ROOT,
        config_note="Transformer(ModelArgs(dim=288, n_layers=6, n_heads=6, n_kv_heads=6, "
        "vocab_size=32000, multiple_of=32, max_seq_len=256, dropout=0.0)); tok_embeddings tied "
        "to output",
        build=_build_llama2c,
        # freqs_cos/freqs_sin are non-persistent (model.py:228-229): absent from the source
        # state_dict, tolerated if a port exports them. attention.mask exists only without
        # SDPA (model.py:113-118).
        exempt_buffers=(r"freqs_cos", r"freqs_sin", r"layers\.\d+\.attention\.mask"),
        tie_groups=(("tok_embeddings.weight", "output.weight"),),  # model.py:224
    ),
    "ast": RepoEntry(
        repo="ast", source_dir=_AST_ROOT,
        config_note="ASTModel(label_dim=35, fstride=10, tstride=10, input_fdim=128, "
        "input_tdim=128, model_size='base384') (Speech Commands V2-35 recipe); constructed with "
        "imagenet_pretrain=False (same names/shapes, no download); timm 0.4.5 from a dependency site",
        build=_build_ast,
        # the 1000-class DeiT heads are constructed but never called (ast_models.py:164-187)
        optional_keys=(r"v\.head\.(weight|bias)", r"v\.head_dist\.(weight|bias)"),
    ),
    # ---- classification: FITS / FITS_WITH_EXEMPTIONS -------------------
    "airbench": RepoEntry(
        repo="airbench", source_dir=_AIRBENCH_ROOT,
        config_note="legacy/airbench94.py make_net(): whitening conv 0.* (24,3,2,2) frozen but "
        "exported, three ConvGroups with BN stats, 7.weight (10,256); 39 entries incl. 6 BN "
        "counters (exempt); export the EMA-collapsed net",
        build=_build_airbench,
    ),
    "deepspeech": RepoEntry(
        repo="deepspeech", source_dir=_DEEPSPEECH_ROOT,
        config_note="DeepSpeech(labels=29, BiDirectionalConfig(lstm, hidden 1024, 5 layers), "
        "precision 32, AdamConfig, SpectConfig); BN running stats required, counters exempt",
        build=_build_deepspeech,
    ),
    "makemore": RepoEntry(
        repo="makemore", source_dir=_MAKEMORE_ROOT,
        config_note="Transformer(ModelConfig(vocab_size=27, block_size=16, n_layer=4, n_head=4, "
        "n_embd=64, n_embd2=64)) (--type transformer on names.txt); untied lm_head",
        build=_build_makemore,
        exempt_buffers=(_CAUSAL_MASK,),  # (1,1,16,16)
    ),
    "nanochat_core": RepoEntry(
        repo="nanochat_core", source_dir=_NANOCHAT_ROOT,
        config_note="speedrun depth 24 at the pin: GPT(GPTConfig(sequence_len=2048, "
        "vocab_size=32768, n_layer=24, n_head=12, n_kv_head=12, n_embd=1536, "
        "window_pattern='SSSL')): MHA, value embeds + ve_gate on the 12 has_ve layers, "
        "resid/x0 lambdas, smear/backout, untied head (1.38B params; "
        "XSTACK_TORCH_LOWMEM=1 builds it in bf16)",
        build=_build_nanochat_core,
    ),
    "nanovlm": RepoEntry(
        repo="nanovlm", source_dir=_NANOVLM_ROOT,
        config_note="VisionLanguageModel(VLMConfig(), load_backbone=False): SigLIP-B/16-224 tower, "
        "SmolLM2-135M decoder (9 q / 3 kv heads), pixel-shuffle MP; lm_tie_weights True",
        build=_build_nanovlm,
        exempt_buffers=(r"decoder\.rotary_embd\.inv_freq",),  # persistent, deterministic
        tie_groups=(("decoder.head.weight", "decoder.token_embedding.weight"),),
    ),
    "patchtst": RepoEntry(
        repo="patchtst", source_dir=_PATCHTST_ROOT,
        config_note="PatchTST Model(traffic pred-96 args: enc_in 862, seq_len 336, e_layers 3, "
        "n_heads 16, d_model 128, d_ff 256, patch 16/8, revin 1 affine 0); BatchNorm1d running "
        "stats required, counters exempt",
        build=_build_patchtst,
        # a requires_grad=False Parameter holding 1/sqrt(d_k); a JAX port folds it into a constant
        optional_keys=(r"model\.backbone\.encoder\.layers\.\d+\.self_attn\.sdp_attn\.scale",),
    ),
    "s4": RepoEntry(
        repo="s4", source_dir=_S4_ROOT,
        config_note="example.py S4Model(d_input=3, d_output=10, d_model=128, n_layers=4, "
        "dropout=0.1, prenorm=False) with S4D(d_state 64): kernel.C is view_as_real (128,32,2)",
        build=_build_s4,
    ),
    "simcse": RepoEntry(
        repo="simcse", source_dir=_SIMCSE_ROOT,
        config_note="BertForCL(BertConfig() bert-base-uncased, pooler cls, temp 0.05, "
        "mlp_only_train): bert.* without pooler, plus mlp.dense; training-model naming, not "
        "the HF-converted form",
        build=_build_simcse,
        exempt_buffers=(r"bert\.embeddings\.position_ids", r"bert\.embeddings\.token_type_ids"),
    ),
    "tabr": RepoEntry(
        repo="tabr", source_dir=_TABR_ROOT,
        config_note="bin/tabr.py Model(covtype 2-lr-evaluation/0.toml: d_main 130, d_multiplier 2, "
        "1+1 blocks, LREmbeddings d 36, 10 num + 44 bin features, 7 classes); 26 tensors",
        build=_build_tabr,
    ),
    "tdmpc2": RepoEntry(
        repo="tdmpc2", source_dir=_TDMPC2_ROOT,
        config_note="WorldModel(config.yaml defaults, walker-run: obs 24, action 6, single task): "
        "5M model, num_q 5 ensemble (leading dim 5), num_bins 101; _Qs.params aliases "
        "_detach_Qs_params; _target_Qs_params is a separate copy",
        build=_build_tdmpc2,
        # TensorDictParams metadata entries (tensordict 0.8.x) and the config-constant buffers
        exempt_buffers=(r".*__batch_size", r".*__device", r"log_std_min", r"log_std_dif"),
        # tensordict >= 0.9 keeps zero-storage template parameters under *.module.* next to
        # the real stacked params; no port exports them
        optional_keys=(r"_Qs\.module\..*", r"_detach_Qs\.module\..*", r"_target_Qs\.module\..*"),
        # _detach_Qs_params.* are buffers aliasing _Qs.params.* (same storage in the source)
        tie_groups=tuple(
            (f"_Qs.params.{i}.{p}", f"_detach_Qs_params.{i}.{p}")
            for i in range(3)
            for p in (("weight", "bias", "ln.weight", "ln.bias") if i < 2 else ("weight", "bias"))
        ),
    ),
    "tensorf": RepoEntry(
        repo="tensorf", source_dir=_TENSORF_ROOT,
        config_note="TensorVMSplit(lego.txt: density_n_comp [16,16,16], appearance_n_comp "
        "[48,48,48], app_dim 27, MLP_Fea view_pe 2 fea_pe 2 featureC 128); gridSize taken from "
        "the export (alpha-mask shrink makes it data-dependent), planes/lines checked for "
        "consistency",
        build_from_npz=_build_tensorf,
        # export metadata the loader reads (like the source's ckpt['kwargs']) and the bit-packed
        # alpha mask the source saves next to the state_dict: tolerated extras
        exempt_buffers=(r"aabb", r"gridSize", r"alphaMask\..*"),
        extra_self_tests=_tensorf_self_tests,
    ),
    "tgn": RepoEntry(
        repo="tgn", source_dir=_TGN_ROOT,
        config_note="TGN(wikipedia 9228x172 nodes, 157475x172 edges; n_layers 1, n_heads 2, memory "
        "172, message 100, graph_attention/identity/last/gru, n_neighbors 10): 28 entries with "
        "shared TimeEncode (load-bearing tie) and the memory state pair (optional)",
        build=_build_tgn,
        # memory.{memory,last_update} are requires_grad=False Parameters = data-dependent
        # state a port may keep outside its parameter tree (26-key export)
        optional_keys=(
            r"memory\.(memory|last_update)", r"memory_updater\.memory\.(memory|last_update)",
        ),
        tie_groups=(
            ("time_encoder.w.weight", "embedding_module.time_encoder.w.weight"),
            ("time_encoder.w.bias", "embedding_module.time_encoder.w.bias"),
            ("memory.memory", "memory_updater.memory.memory"),
            ("memory.last_update", "memory_updater.memory.last_update"),
        ),
    ),
    "x_transformers": RepoEntry(
        repo="x_transformers", source_dir=_XTR_ROOT,
        config_note="train_enwik8.py: TransformerWrapper(num_tokens=256, max_seq_len=1024, "
        "Decoder(dim=512, depth=6, heads=8, rotary_pos_emb=False, polar_pos_emb=True, "
        "pre_and_post_norm=True)); one shared polar learned_bias; untied to_logits",
        build=_build_x_transformers,
        exempt_buffers=(r"attn_layers\.polar_pos_emb\.inv_freq",),
        extra_strip_prefixes=("net.",),  # AutoregressiveWrapper
    ),
}


