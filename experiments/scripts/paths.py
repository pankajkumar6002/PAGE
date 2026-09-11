"""Path resolution for the revision-round analysis scripts (R1 + R2).

The two revision rounds were originally developed in sibling checkouts
(`new-exp_page-kv/` and `new-exp-page-kv-r2/`), each with its own `paths.py`.
Both rounds now live here, alongside the original scripts, so this module is
the union of the two: same env-var contract, same helpers, one set of roots.

The original scripts in this directory hardcode the run machine's paths
(`/home/smlab/projects/eff-nn/...`) and do not execute here. The revision-round
scripts resolve everything relative to this checkout instead, with env
overrides for other machines.

  PAGE_RESULTS   per-input logs from the original released runs
                 (default: ../results, i.e. page-kv/experiments/results)
  PAGE_DATA      where the revision rounds' GPU-run jsonl are read from
                 (default: ../results — same directory; the rounds' outputs
                 were merged into it)
  PAGE_OUT       where regenerated summaries are written
                 (default: ../results)
  PAGE_SRC       the original scripts, for importing the canonical runner
                 (default: this directory)

RESULTS, DATA and OUT are kept as separate names even though they now default
to the same directory: redirecting writes with PAGE_OUT must not silently
redirect reads, which once made two analyses report "no complete cells" when
they simply were not looking where the data lives.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_PAGEKV = os.path.abspath(os.path.join(_EXPERIMENTS, ".."))

RESULTS = os.path.abspath(os.environ.get(
    "PAGE_RESULTS", os.path.join(_EXPERIMENTS, "results")))
DATA = os.path.abspath(os.environ.get(
    "PAGE_DATA", os.path.join(_EXPERIMENTS, "results")))
OUT = os.path.abspath(os.environ.get(
    "PAGE_OUT", os.path.join(_EXPERIMENTS, "results")))
SRC = os.path.abspath(os.environ.get("PAGE_SRC", _HERE))

TAU = 0.07
MK3 = "niah_multikey_3"
TASKS = ["niah_multikey_3", "vt", "fwe", "qa_1"]

# The 4x4 headline matrix of tab:matrix. R1 scripts unpack pairs; R2 scripts
# also need the HF model id, so they read MODELS_HF instead.
MODELS = [
    ("Qwen2.5-1.5B", "qwen15b"),
    ("Qwen2.5-3B", "qwen3b"),
    ("Qwen2.5-14B", "qwen14b"),
    ("Mistral-7B", "mistral7b"),
]
MODELS_HF = [
    ("Qwen2.5-1.5B", "qwen15b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "qwen3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen2.5-14B", "qwen14b", "Qwen/Qwen2.5-14B-Instruct"),
    ("Mistral-7B", "mistral7b", "mistralai/Mistral-7B-Instruct-v0.3"),
]
POLICIES = ["snapkv", "h2o", "streamingllm", "pyramidkv"]

# Executed at tau=0.04, reported at tau=0.07 by post-hoc re-evaluation
# (tab:matrix caption + App. app:repro).
RECONSTRUCTED = {("snapkv", "qwen3b")}

PREREG = os.path.abspath(os.path.join(_PAGEKV, "preregistration"))
PREREG_SHA = "c112a20265464a4f4d606c156993ff6622392e49b4ba285dc967f2bc942b9bf3"


def cell_path(slug, policy):
    """Log file for one (model, base-evictor) cell of the 4x4 matrix."""
    if policy == "snapkv":
        return os.path.join(RESULTS, f"gated_4k_{slug}.jsonl")
    return os.path.join(RESULTS, f"gated_{policy}_{slug}_4k.jsonl")


# R2 spelling of the same lookup.
def released_cell(slug, policy="snapkv"):
    """Log file for one cell of the released 4x4 matrix."""
    return cell_path(slug, policy)


def adakv_cell(slug):
    """R2's per-head Ada-KV run for one model."""
    return os.path.join(DATA, f"adakv_4k_{slug}.jsonl")


def require(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing input log: {path}\n"
                         f"set PAGE_RESULTS/PAGE_DATA to the directory holding it")
    return path


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def verify_prereg():
    """The per-head Ada-KV pre-registration must be intact and unmodified since hashing."""
    import hashlib
    p = os.path.join(PREREG, "adakv_perhead_prereg.md")
    if not os.path.exists(p):
        raise SystemExit(f"pre-registration missing: {p}")
    # Only the predictions are frozen. Everything from the "## Addendum"
    # heading onward is measurement notes recorded during setup, which must be
    # appendable without invalidating the commitment made before the run.
    raw = open(p, "rb").read()
    i = raw.find(b"## Addendum")
    if i != -1:
        raw = raw[:i - 6]          # drop the "\n---\n\n" separator too
    got = hashlib.sha256(raw).hexdigest()
    if got != PREREG_SHA:
        raise SystemExit(
            f"pre-registration hash mismatch\n  expected {PREREG_SHA}\n"
            f"  got      {got}\nThe predictions were edited after registration.")
    return got
