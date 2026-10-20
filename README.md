# Tenant-ive Plans

**Team Carrier Pigeon:** Abbie Terhoeve, Matthew Stafford, Peter Vo

Predicting the probability that a Gated Rentals tenant moves out in a given week, from tenant features such as income, family size, and commute. Full proposal: [docs/project-proposal-initial.md](docs/project-proposal-initial.md).

- **Data:** anonymized extract covering two properties (Willis Lakes, Crosby Heights): 455 households, 7,566 billing cycles. Ships as `Gated_Rentals_Dataset.zip`.
- **Model:** a feed-forward neural network that outputs, for each household-week, the probability the household leaves that week (a discrete-time hazard model).
- **Baselines:** a Kaplan-Meier survival curve (no tenant features) and a logistic regression on the same inputs as the neural network.
- **Evaluation:** calibration and Brier score at 26 and 52 weeks, scored with bootstrap sampling and grouped k-fold cross-validation; an ablation with and without commute features; a rerun on the highest-confidence subset of the data; and fairness checks.

## Setup

You need **Python 3.13 or newer** (`python3 --version` to check).

### 1. Create and activate a virtual environment

From the repo root:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Activate it again every time you open a new terminal.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Linux only:** the default `torch` download bundles CUDA and is several GB. We don't need a GPU, so install the CPU build first, then the rest:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Unpack the data

```bash
python scripts/set-up-repo.py
```

This reads `Gated_Rentals_Dataset.zip`, checks every file against the provider's checksums, and copies the files we use into `data/`:

```
data/
├── raw/              residents.csv, weekly_panel.csv     # model inputs, never edit by hand
├── processed/        (empty)                              # output of preprocessing/preprocess.py
├── reference/        km_survival.csv, sensitivity_checks.csv  # provider's results, for sanity checks
└── provider-docs/    DATA_DICTIONARY.md, README.md        # provider's documentation
```

`data/` is gitignored. The zip is the source of truth: if anything in `data/raw/` gets changed, rerun the script to restore it. If it fails with a checksum mismatch, the zip is corrupted; pull a fresh copy.

## Repo layout

```
Gated_Rentals_Dataset.zip   # the dataset, as delivered by the provider
scripts/set-up-repo.py      # unpacks the zip into data/
preprocessing/              # data/raw -> data/processed
docs/
├── README.md               # proposed architecture and experiment flow
├── OPEN-ISSUES.md          # open questions and decisions to make
└── architecture/
    └── data-dictionary.md  # which columns we use, which we don't, and why
```

## Adding a dependency

Install it, then add **only that package** to `requirements.txt` as `<package>==<version>`:

```bash
pip install <package>
pip show <package>     # the Version: line is what goes after ==
```

Don't paste the whole `pip freeze` output into `requirements.txt`. Pip resolves the sub-dependencies itself.
