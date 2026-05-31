# AGENTS.md — dicoding-membangun-sistem-ML

**Project:** Dicoding submission — "Membangun Sistem Machine Learning"  
**Task:** NSL-KDD intrusion detection (binary classification: normal → 0, attack → 1)  
**Model:** DecisionTreeClassifier  
**Default branch:** `Main` (remote default is `Main`; both `Main` and `master` exist locally)

## Structure

```
Eksperimen_SML_Rudy_Eko/
├── dataset-nsl-kdd/             # NSL-KDD raw data (.arff, .txt)
├── preprocessing/
│   ├── automate_preprocessing.py   # RobustScaler + one-hot encode + outcome→binary
│   └── Template_Eksperimen_MSML.ipynb  # EDA + preprocessing notebook
└── modelling/
    ├── modelling.py               # DecisionTreeClassifier training, evaluation, MLflow logging
    └── preprocessed_kdd.csv       # preprocessed output (125973 rows, 124 columns)
mlruns/                            # MLflow tracking dir (file-based)
requirements.txt                   # pinned deps from pip freeze
```

## Environment

- Python 3.13, venv at `.venv/`
- Activate: `source .venv/bin/activate` (Linux) or `.venv\Scripts\Activate.ps1` (Windows)
- Key packages: scikit-learn 1.8.0, pandas, numpy, matplotlib, seaborn, mlflow 3.12.0

## Workflow

1. **Preprocessing:** `Template_Eksperimen_MSML.ipynb` → calls `automate_preprocessing.preprocess()` → writes `preprocessed_kdd.csv`
2. **Modelling:** `modelling.py` reads `preprocessed_kdd.csv` (its sibling), splits 80/20 (random_state=42), trains DecisionTreeClassifier, logs to MLflow, saves `plot_feature_importance.png` and `plot_confusion_matrix.png`

### Preprocessing details (`automate_preprocessing.py`)
- Drop numeric columns → `RobustScaler` → reinsert
- `outcome`: `normal` → 0, everything else → 1
- One-hot encode: `protocol_type`, `service`, `flag`
- Keep as-is: `land`, `logged_in`, `is_host_login`, `is_guest_login`, `level`

### Modelling details (`modelling.py`)
- Loads `preprocessed_kdd.csv` via `os.path.dirname(os.path.abspath(__file__))` (sibling path)
- MLflow experiment: `"KDD Cyber Attack Classify"`; view UI: `mlflow ui --port 5000`
- Generates plots: `plot_feature_importance.png`, `plot_confusion_matrix.png` (saved alongside script)

## Commands

```bash
# run the full pipeline
source .venv/bin/activate
python Eksperimen_SML_Rudy_Eko/modelling/modelling.py

# view MLflow UI
mlflow ui --port 5000
```

## Commit style

`[ADD]` / `[FIX]` / `[EDIT]` prefix in Indonesian or English. No conventional commits.

## Git notes

- Remote: `https://github.com/rudyekoprasetya/dicoding-membangun-sistem-ML.git`
- `.gitignore` excludes: `.venv/`, `__pycache__/`, `.vscode/`, `.idea/`, `.env`, `*.log`, `.qodo/`
