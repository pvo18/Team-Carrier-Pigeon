# Move-Out Study Commented Code

This package contains the project-owned Python code used to prepare, train,
validate, compare, and visualize the weekly move-out and survival analyses.

Documentation standard applied:

- Every Python file begins with a structured summary covering its purpose,
  inputs, outputs, workflow, and important methodological constraint.
- Every function and method has an explanatory docstring.
- Inline comments explain leakage prevention, censoring logic, fold isolation,
  controlled hyperparameter changes, artifact recovery, and plot construction.
- Comments describe the implementation without changing model settings,
  calculations, input data, or previously reported results.

Package contents:

- `analysis/complete_analysis.py`: final plots and summary-table generation.
- `analysis/hyperparameter_study.py`: controlled one-factor hyperparameter trials.
- `hyperparameter_studies/classification/`: standalone, self-contained weekly
  classification trial code, fixed inputs, completed results, figures, and
  explanatory Word document.
- `hyperparameter_studies/survival/`: standalone, self-contained Classical Cox
  and Neural CoxPH trial code, fixed inputs, all 900 completed evaluations,
  figures, and explanatory Word document.
- `orchestration/`: snapshot preparation, training dispatch, recovery,
  aggregation, qualification, reporting, and packaging scripts.
- `weekly_moveout/`: reusable modeling package, configurations, and tests.
- `plots_and_tables/`: the eight final figures and their supporting result tables.

Verification performed after commenting:

- Python compilation: passed for all included source files.
- Automated modeling tests: 20 passed.
- Final analysis regeneration: passed; all 30 hyperparameter trials were read.
- Documentation audit: 29 Python files and 97 functions/methods checked, with
  zero missing module summaries and zero missing function docstrings.

## November 24 packaging update

The hyperparameter work is now separated by analysis question. Both standalone
sets use relocatable paths and include the source and data dependencies needed
to reproduce their trials. The classification runner no longer depends on the
original workspace, and the survival set now includes the dependencies that
were absent from its earlier results-only archive. All completed numerical
results are unchanged.

