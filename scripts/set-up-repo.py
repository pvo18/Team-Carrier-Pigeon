"""
Unpacks Gated_Rentals_Dataset.zip into the data/ layout the rest of the repo expects.

Run from anywhere after cloning:  python scripts/set-up-repo.py

Only copies the files we actually use. Everything else stays in the zip.
Every file is checked against the provider's SHA256SUMS before it's written,
so if the zip is corrupted or someone edits it, this fails loudly instead of
giving you bad data.

data/raw/ is a copy of the zip, so rerunning this overwrites it. Never edit
files in data/raw by hand; do the cleaning in preprocessing/preprocess.py
and write the results to data/processed/.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in scripts/
ZIP_PATH = REPO_ROOT / "Gated_Rentals_Dataset.zip"
DATA_DIR = REPO_ROOT / "data"
ZIP_PREFIX = "willis-crosby-cohort-clv-20260915/"

# path inside the zip (after ZIP_PREFIX) -> folder under data/
FILES_TO_EXTRACT = {
    # model inputs: the weekly rows + label, and the signup-time features
    "data/weekly_panel.csv": "raw",
    "data/residents.csv": "raw",
    # provider's own results, only for sanity-checking ours (fit on ALL the data,
    # so don't compare them directly against cross-validated scores)
    "data/km_survival.csv": "reference",
    "data/sensitivity_checks.csv": "reference",
    # provider docs
    "DATA_DICTIONARY.md": "provider-docs",
    "README.md": "provider-docs",
}

EMPTY_DIRS = ["processed"]


def read_checksums(archive: zipfile.ZipFile) -> dict[str, str]:
    """Parses SHA256SUMS ('<hash>  <path>' per line) into {path: hash}."""
    text = archive.read(ZIP_PREFIX + "SHA256SUMS").decode("utf-8")
    checksums = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        expected_hash, path = line.split(maxsplit=1)
        checksums[path.strip()] = expected_hash.lower()
    return checksums


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(f"Can't find {ZIP_PATH.name} in {REPO_ROOT}. Did you clone the whole repo?")

    # verify everything before writing anything, so a bad zip never leaves data/ half-updated
    verified: dict[str, bytes] = {}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        checksums = read_checksums(archive)
        for member in FILES_TO_EXTRACT:
            contents = archive.read(ZIP_PREFIX + member)
            actual_hash = hashlib.sha256(contents).hexdigest()
            if actual_hash != checksums[member]:
                raise SystemExit(
                    f"Checksum mismatch for {member}: expected {checksums[member]}, got {actual_hash}. "
                    "The zip is corrupted or was modified; re-download it."
                )
            verified[member] = contents

    for member, folder in FILES_TO_EXTRACT.items():
        destination = DATA_DIR / folder / Path(member).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(verified[member])
        print(f"OK  {member}  ->  {destination.relative_to(REPO_ROOT)}")

    for folder in EMPTY_DIRS:
        (DATA_DIR / folder).mkdir(parents=True, exist_ok=True)

    print(f"\nDone. {len(FILES_TO_EXTRACT)} files verified and written under data/.")


if __name__ == "__main__":
    main()
