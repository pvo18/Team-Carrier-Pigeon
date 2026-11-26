#!/usr/bin/env python3
"""Compatibility launcher for the separated classification trial package.

Purpose: Preserve the original command location while directing the user to
the now self-contained, heavily commented classification experiment runner.
Inputs/outputs: Command-line arguments and trial artifacts are handled by the
target runner without alteration.
Method note: The executable study is stored under ``hyperparameter_studies``
so classification and survival code/results can be submitted independently.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    """Execute the standalone classification runner as the main program."""
    # Resolve from the combined package root rather than the caller's current
    # directory, allowing the ZIP to be extracted anywhere on the computer.
    target = (
        Path(__file__).resolve().parents[1]
        / "hyperparameter_studies"
        / "classification"
        / "run_classification_hyperparameter_trial.py"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
