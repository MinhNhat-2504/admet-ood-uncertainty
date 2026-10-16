"""Download the OpenADMET-ExpansionRx dataset and verify its published checksums.

Refuses to write a file whose MD5 differs from the Zenodo record, so an experiment can
never silently run on a changed snapshot.
"""

from __future__ import annotations

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path
from urllib.request import urlopen

RECORD = "https://zenodo.org/records/21504733"
DOI = "10.5281/zenodo.21504733"
LICENCE = "CC-BY-4.0"
# MD5 values as published on the Zenodo record page.
FILES = {
    "README.md": "dd24ad6e8af26790c4b6d5ba6992c4c7",
    "expansion_data_train.csv": "f9134d06cc565826a8150f577d380557",
    "expansion_data_test.csv": "93a0ab4abdecbd108b1a928ddb70bd59",
    "expansion_data_raw.csv": "27fae3d509fb1d9fd97cc94a4c9cccec",
}
DEFAULT_DIR = Path("data/raw/expansionrx")


def fetch(name: str, destination: Path, timeout: int = 120) -> bytes:
    with urlopen(f"{RECORD}/files/{name}?download=1", timeout=timeout) as response:
        payload = response.read()
    digest = md5(payload).hexdigest()
    if digest != FILES[name]:
        raise ValueError(f"{name}: MD5 {digest} does not match published {FILES[name]}")
    destination.write_bytes(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    receipt = {"source": RECORD, "doi": DOI, "licence": LICENCE, "files": {}}
    for name in FILES:
        path = args.output_dir / name
        if path.is_file() and md5(path.read_bytes()).hexdigest() == FILES[name]:
            payload = path.read_bytes()
            status = "already_present"
        else:
            payload = fetch(name, path)
            status = "downloaded"
        receipt["files"][name] = {"status": status, "bytes": len(payload),
                                  "md5": FILES[name],
                                  "sha256": sha256(payload).hexdigest()}
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "files": {k: v["status"] for k, v in receipt["files"].items()}}))


if __name__ == "__main__":
    main()
