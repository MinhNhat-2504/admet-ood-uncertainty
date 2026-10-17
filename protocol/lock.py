"""Freeze and verify the preregistered inputs before any model is fitted.

`create` records SHA-256 digests of the protocol, the analysis plan, the frozen data
files and every source module that can influence a result, plus the runtime fingerprint.
`verify` recomputes them; scoring scripts call it and refuse to report if anything moved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "protocol" / "LOCK.json"
STATUS = "preregistered_before_any_model_fit"

PROTOCOL_FILES = ("protocol/PROTOCOL.md", "protocol/ANALYSIS_PLAN.md", "protocol/lock.py")
DATA_FILES = (
    "data/raw/expansionrx/expansion_data_train.csv",
    "data/raw/expansionrx/expansion_data_test.csv",
)
# Every module whose behaviour can change a reported number.
SOURCE_GLOB = "src/admetood/*.py"

# Prespecified constants, duplicated here so a silent edit to the code is detectable.
SETTINGS = {
    "nominal_coverage": 0.90,
    "equivalence_margin_pp": 5.0,
    "test_fraction": 0.30,
    "model_seeds": [0, 1, 2, 3, 4],
    "split_seed": 20261017,
    "bootstrap_seed": 20261018,
    "bootstrap_resamples": 10000,
    "simulated_splits": ["random", "scaffold", "butina", "frontier", "size"],
    "uncertainty_methods": ["ensemble", "quantile", "split_conformal",
                            "mondrian_conformal", "distance"],
    "eligibility": {"min_train_rows": 500, "min_test_rows": 250},
    "target_absolute_error": 0.5,
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def digests(root: Path = ROOT) -> dict[str, str]:
    """Digest every frozen input; missing files fail loudly rather than being skipped."""
    paths = [root / name for name in PROTOCOL_FILES + DATA_FILES]
    paths += sorted(root.glob(SOURCE_GLOB))
    missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot lock; missing: {', '.join(missing)}")
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def runtime() -> dict[str, str]:
    import numpy, pandas, rdkit, sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "rdkit": rdkit.__version__,
        "scikit-learn": sklearn.__version__,
    }


def create(root: Path = ROOT, path: Path = LOCK) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} exists; amend the plan instead of relocking")
    payload = {
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
        "digests": digests(root),
        "runtime": runtime(),
        "note": ("Created before any model was fitted. Source digests cover modules that "
                 "existed at lock time; modules added later are recorded by verify()."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify(root: Path = ROOT, path: Path = LOCK, strict_sources: bool = False) -> dict:
    """Return the lock, raising if a frozen input changed.

    Data, protocol and settings must match exactly. Source modules are checked for
    modification; new modules are allowed unless `strict_sources` is set, because
    analysis code is written after the lock by design.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No lock at {path}; run `python protocol/lock.py create`")
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked.get("status") != STATUS:
        raise ValueError("Unrecognized lock status")
    if locked["settings"] != SETTINGS:
        raise ValueError("Prespecified settings differ from the lock")

    current = digests(root)
    changed = [name for name, value in locked["digests"].items()
               if current.get(name) != value]
    if changed:
        raise ValueError("Frozen inputs changed since lock: " + ", ".join(sorted(changed)))
    if strict_sources:
        added = sorted(set(current) - set(locked["digests"]))
        if added:
            raise ValueError("New source modules since lock: " + ", ".join(added))
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    if args.action == "create":
        payload = create()
        print(json.dumps({"lock": str(LOCK), "created_utc": payload["created_utc"],
                          "files_locked": len(payload["digests"])}, indent=2))
        return
    locked = verify()
    print(json.dumps({"lock": str(LOCK), "status": locked["status"],
                      "created_utc": locked["created_utc"],
                      "files_verified": len(locked["digests"])}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
