# AGENTS.md — dicoding-smsml

**Project:** Dicoding submission — "Membangun Sistem Machine Learning"  
**Task:** NSL-KDD intrusion detection (binary classification: normal vs attack)  
**Model:** KNeighborsClassifier  
**Default branch:** `Main` (both `Main` and `master` exist; remote default is `Main`)

## Structure

```
Eksperimen_SML_Rudy_Eko/
├── dataset-nsl-kdd/       # NSL-KDD raw data (.arff, .txt)
├── preprocessing/
│   ├── automate_preprocessing.py    # RobustScaler + one-hot encode + outcome→binary
│   └── Template_Eksperimen_MSML.ipynb  # EDA + preprocessing notebook
└── modelling/
    └── modelling.ipynb               # KNN modelling notebook
preprocessed_kdd.csv                  # preprocessed output
```

## Environment

- Python 3.13, venv at `.venv/`
- Key packages: scikit-learn 1.8.0, pandas, numpy, matplotlib, seaborn, mlflow
- Activate: `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate`
- No requirements.txt — use `pip list` to inspect deps

## Workflow

1. **Preprocessing:** run `Template_Eksperimen_MSML.ipynb` → calls `automate_preprocessing.preprocess()` → writes `preprocessed_kdd.csv`
2. **Modelling:** `modelling.ipynb` reads `preprocessed_kdd.csv`, trains KNN with 80/20 split (random_state=42), uses mlflow for tracking

### Preprocessing details (`automate_preprocessing.py`)
- Drop-and-replace numeric columns with `RobustScaler`
- `outcome` → binary: `normal` → 0, anything else → 1
- One-hot encode: `protocol_type`, `service`, `flag`
- Kept as-is: `land`, `logged_in`, `is_host_login`, `is_guest_login`, `level`

## Commit style

`[ADD]` / `[FIX]` / `[SOMETHING]` prefix in Indonesian or English. No conventional commits.

## Git notes

- Remote: `https://github.com/rudyekoprasetya/dicoding-membangun-sistem-ML.git`
- `.gitignore` excludes: `.venv/`, `__pycache__/`, `.vscode/`, `.idea/`, `.env`, `*.log`, `.qodo/`
