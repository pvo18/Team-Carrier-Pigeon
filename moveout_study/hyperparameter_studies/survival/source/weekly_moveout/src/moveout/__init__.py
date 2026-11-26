"""Move-out study source summary.
Purpose: Define the public identity and version of the weekly move-out modeling package.
Inputs: The package supports weekly classification and censor-aware survival prediction using shared prepared data.
Outputs: Importing this module exposes only package metadata and has no training or file-system side effects.
Workflow: The version is written into run manifests for reproducibility.
Method note: Model, data, metric, and pipeline implementations live in sibling modules.
"""

__version__ = "0.2.0"
