# PyTorch → JAX on TPU: whole-repo transfer

Port the fixed-commit PyTorch repository under `source_torch/` entirely to a
working JAX/Flax/Optax repository, and make it work at real scale on the TPU machine
described in `TPU_ACCESS.md` — a clean machine: set it up yourself. You have
full access to that machine to develop, run, and test your port. The
deliverable is the working repository itself: produce a faithful port and
run it the way the source does.

Deliverables at this workspace root:

- `setup_env.sh` — prepares a worker of the TPU machine from clean state,
  including any data the repository's pipeline needs, obtained and prepared
  the way the source repository does. It is sourced before the entry point
  runs.
- `run_train.py --seed S --out DIR --metrics PATH` — runs the repository's
  training pipelines the way the repository runs them, at their own scale,
  and writes the numbers the repository itself reports to the metrics JSON. It
  checkpoints to `--out` as it trains and resumes from an existing
  checkpoint in `--out`.
- `export_torch.py --ckpt DIR --out PATH` — writes the trained weights as
  an `.npz` of numpy arrays keyed by the source model's own parameter names
  (exact names and shapes).
- The graded run trains the HSTU-large configuration on the MovieLens-20M dataset.

Your submission is collected from this workspace and executed on a fresh
machine of the same type. It is judged on behavior that is fundamental to
the source repository, not all of which is spelled out here.

Rules:

- Do not import or call PyTorch anywhere in your submission at runtime.
- Data, library, and package downloads are allowed; do not fetch, clone, or
  copy model or porting code from the internet.

## Goal

Keep working until your submission passes grading.
