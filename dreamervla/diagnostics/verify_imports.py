"""CPU-only import & version validator for DreamerVLA.

Run this on the (GPU-less) machine where you build the conda env, BEFORE shipping
that env to a GPU box:

    python -m dreamervla.diagnostics.verify_imports

It needs NO GPU and loads NO checkpoint. Unlike a plain "import every module"
sweep, it also reproduces the *deferred* imports that only run inside
``OpenVLAOFTPolicy.__init__`` (e.g. ``from peft import LoraConfig, get_peft_model``)
-- the exact chain that crashes a GPU run with::

    ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'

That error means peft drifted above the pinned ``peft==0.11.0`` (newer peft does
``from transformers import ... EncoderDecoderCache``, which the pinned
transformers 4.40.1 does not have). Check 3 below catches it deterministically on
a CPU host, so it can never reach the GPU box silently.

Exit code 0 = safe to ship this env. Non-zero = at least one blocking problem.
"""

from __future__ import annotations

import importlib
import importlib.metadata as ilm
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# Optional, host-/hardware-specific deps. A dreamervla module that fails to
# import ONLY because one of these is absent on this (CPU/build) host is SKIPPED,
# not failed -- it will be present on the GPU box.
OPTIONAL_HOST_DEPS = {
    "libero",
    "robosuite",
    "robosuite_task_zoo",
    "robomimic",
    "mimicgen",
    "mujoco",
    "mujoco_py",
    "egl",
    "ray",
    "flash_attn",
    "deepspeed",
    "bitsandbytes",
    "apex",
    "opensora",
}

# Pin drifts here are HARD failures (proven to break the runtime). Everything
# else in requirements.txt is reported but only warns.
CRITICAL_PINS = {"peft"}

# Deprecated tree kept for reference: its import failures are reported but never
# block the gate.
NONBLOCKING_PREFIXES = ("dreamervla.legacy",)

# The exact deferred imports OpenVLAOFTPolicy.__init__ runs (openvla_oft_policy.py
# lines 83-98 + 147), grouped so each group's failure is reported independently.
LAZY_IMPORT_GROUPS = [
    (
        "peft LoRA  <-- the EncoderDecoderCache crash site",
        "from peft import LoraConfig, get_peft_model",
    ),
    (
        "prismatic OpenVLA config/model/processor",
        "from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig\n"
        "from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction\n"
        "from prismatic.extern.hf.processing_prismatic import (\n"
        "    PrismaticImageProcessor, PrismaticProcessor,\n"
        ")",
    ),
    (
        "prismatic projectors/tokenizer/constants/action-head",
        "from prismatic.models.projectors import ProprioProjector\n"
        "from prismatic.vla.action_tokenizer import ActionTokenizer\n"
        "from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM\n"
        "from prismatic.models.action_heads import L1RegressionActionHead",
    ),
    (
        "transformers Auto* (Vision2Seq etc.)",
        "from transformers import (\n"
        "    AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,\n"
        ")",
    ),
]


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_pins(path: Path) -> dict[str, str]:
    """Parse ``name==version`` lines from requirements.txt (the source of truth)."""
    pins: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.split("[", 1)[0].strip()  # drop extras, e.g. imageio[ffmpeg]
        version = version.strip()
        if name and version:
            pins[name] = version
    return pins


def _installed_version(dist_name: str) -> str | None:
    try:
        return ilm.version(dist_name)
    except ilm.PackageNotFoundError:
        return None


def check_versions(pins: dict[str, str]) -> list[str]:
    failures: list[str] = []
    print(f"{'package':24} {'pinned':12} {'installed':14} status")
    for name, want in sorted(pins.items(), key=lambda kv: kv[0].lower()):
        have = _installed_version(name)
        critical = name.lower() in CRITICAL_PINS
        if have is None:
            status, ok = "NOT INSTALLED", False
        elif have == want:
            status, ok = "ok", True
        else:
            status, ok = "MISMATCH", False
        flag = "" if ok else ("  <-- HARD FAIL" if critical else "  (warn)")
        print(f"{name:24} {want:12} {str(have):14} {status}{flag}")
        if not ok and critical:
            failures.append(f"version: {name} pinned=={want} but installed={have}")
    return failures


def check_transformers_fork() -> list[str]:
    """transformers is not pinned in requirements.txt; the real contract is the
    moojink OpenVLA-OFT fork (vanilla reports the same 4.40.1 but gives 0%
    garbage actions). Mirror scripts/install/60_verify.sh's structural check."""
    try:
        import transformers
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        traceback.print_exc()
        return [f"`import transformers` failed: {type(exc).__name__}: {exc}"]

    print(f"transformers {transformers.__version__} @ {transformers.__file__}")
    oft_present = (PROJECT_ROOT / "third_party" / "openvla-oft").is_dir() or (
        PROJECT_ROOT / "dreamervla" / "models" / "embodiment" / "openvla_oft"
    ).is_dir()
    if not oft_present:
        print("OpenVLA-OFT tree absent; skipping fork check.")
        return []

    llama = Path(transformers.__file__).parent / "models" / "llama" / "modeling_llama.py"
    try:
        src = llama.read_text()
    except OSError:
        print(f"could not read {llama}; skipping fork check.")
        return []
    is_fork = ("is_causal=False" in src) and ("Moo Jin" in src)
    print(f"moojink OFT fork active: {is_fork}")
    if not is_fork:
        return [
            "transformers is VANILLA, not the moojink OpenVLA-OFT fork "
            "(both report 4.40.1). OFT inference -> 0% garbage actions. "
            "Re-run scripts/install/40_third_party.sh; see SETUP.md."
        ]
    return []


def check_lazy_runtime_imports() -> list[str]:
    """Reproduce OpenVLAOFTPolicy.__init__'s deferred imports without building any
    model. This is the authoritative importability test for the OFT runtime."""
    try:
        from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

        root = ensure_openvla_oft_on_path()
        print(f"OpenVLA-OFT tree on sys.path: {root}")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return [f"ensure_openvla_oft_on_path() failed: {type(exc).__name__}: {exc}"]

    failures: list[str] = []
    namespace: dict[str, object] = {}
    for label, code in LAZY_IMPORT_GROUPS:
        try:
            exec(code, namespace)  # noqa: S102 - validator intentionally runs the real imports
            print(f"  OK   {label}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append(f"lazy import [{label}]: {type(exc).__name__}: {exc}")
    return failures


def _iter_dreamervla_modules() -> list[str]:
    pkg_root = PROJECT_ROOT / "dreamervla"
    names: set[str] = set()
    for path in pkg_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue  # importing a __main__ shim can execute a script body
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
    return sorted(names)


def _optional_dep_missing(exc: BaseException) -> str | None:
    """If the import failed ONLY because an optional host dep is absent, return
    its name; otherwise None (a real failure)."""
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        if exc.name.split(".")[0] in OPTIONAL_HOST_DEPS:
            return exc.name
    return None


def _is_standalone_script(module_name: str, exc: BaseException) -> bool:
    """HF-style conversion scripts use implicit-relative imports (e.g.
    ``import configuration_chameleon``) that only resolve when the file is run
    directly. Detect: the missing top-level module is actually a sibling .py
    file (or subpackage dir) next to the module we tried to import -- so this is
    a standalone script, not importable as a package, and not an env problem."""
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return False
    top = exc.name.split(".")[0]
    mod_path = PROJECT_ROOT.joinpath(*module_name.split("."))
    pkg_dir = mod_path.parent if mod_path.with_suffix(".py").exists() else mod_path
    return (pkg_dir / f"{top}.py").exists() or (pkg_dir / top).is_dir()


def walk_import_dreamervla() -> list[str]:
    names = _iter_dreamervla_modules()
    total = len(names)
    ok = 0
    skipped: list[tuple[str, str]] = []
    scripts: list[str] = []
    nonblocking: list[str] = []
    failed: list[str] = []
    for i, name in enumerate(names, 1):
        if i % 50 == 0:
            print(f"  ... {i}/{total}", flush=True)
        try:
            importlib.import_module(name)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            missing = _optional_dep_missing(exc)
            if missing is not None:
                skipped.append((name, missing))
            elif _is_standalone_script(name, exc):
                scripts.append(f"{name}: {type(exc).__name__}: {exc}")
            elif name.startswith(NONBLOCKING_PREFIXES):
                nonblocking.append(f"{name}: {type(exc).__name__}: {exc}")
            else:
                failed.append(f"{name}: {type(exc).__name__}: {exc}")

    print(f"imported OK: {ok}/{total}")
    if skipped:
        print(f"skipped (optional host dep absent on this machine): {len(skipped)}")
        for name, dep in skipped:
            print(f"  SKIP {name}  (needs {dep})")
    if scripts:
        print(f"standalone scripts (implicit-relative imports, non-blocking): {len(scripts)}")
        for item in scripts:
            print(f"  skip {item}")
    if nonblocking:
        print(f"legacy import failures (non-blocking): {len(nonblocking)}")
        for item in nonblocking:
            print(f"  warn {item}")
    if failed:
        print(f"REAL import failures: {len(failed)}")
        for item in failed:
            print(f"  FAIL {item}")
    return failed


def main() -> int:
    print("DreamerVLA CPU-only import & version validator (no GPU, no checkpoints)")
    print(f"repo: {PROJECT_ROOT}")

    problems: list[str] = []
    pins = load_pins(REQUIREMENTS)

    _hr("1/4  version pins (requirements.txt)")
    problems += check_versions(pins)

    _hr("2/4  transformers + moojink OpenVLA-OFT fork")
    problems += check_transformers_fork()

    _hr("3/4  runtime lazy imports (OpenVLAOFTPolicy.__init__ chain) -- authoritative")
    problems += check_lazy_runtime_imports()

    _hr("4/4  import every dreamervla.* module")
    problems += walk_import_dreamervla()

    _hr("SUMMARY")
    if problems:
        print(f"FAILED -- {len(problems)} blocking problem(s):")
        for item in problems:
            print(f"  - {item}")
        if any(item.startswith("version: peft") for item in problems) or any(
            "EncoderDecoderCache" in item for item in problems
        ):
            print(f'\nLikely fix (peft drift):\n    pip install "peft=={pins.get("peft", "0.11.0")}"')
        print("\nDO NOT ship this env to the GPU box until the above are resolved.")
        return 1

    print("PASS -- all import & version checks succeeded.")
    print("This env is safe to ship to the GPU box.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
