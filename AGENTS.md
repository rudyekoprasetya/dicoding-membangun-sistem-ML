# AGENTS.md — dicoding-membangun-sistem-ML

**Task:** NSL-KDD intrusion detection (binary: normal→0, attack→1)  
**Model:** GaussianNB (Naive Bayes) — *not* DecisionTreeClassifier  
**Default branch:** `Main` (only `Main` exists; no `master`)

## Structure

```
Eksperimen_SML_Rudy_Eko/
├── dataset-nsl-kdd/                 # NSL-KDD raw (.arff, .txt)
├── preprocessing/
│   ├── automate_preprocessing.py    # RobustScaler + one-hot + outcome→binary
│   └── Template_Eksperimen_MSML.ipynb
├── modelling/
│   ├── modelling.py                 # GaussianNB, MLflow, no plots saved
│   └── preprocessed_kdd.csv         # 125973 rows, 124 cols
└── Workflow-CI/
    └── MLproject/
        ├── MLproject                # MLflow project entrypoint
        ├── conda.yaml               # Conda env for mlflow run
        └── modelling.py             # Tuning via RandomizedSearchCV + plots + model.pkl
mlruns/                              # MLflow tracking dir
home/                                # (gitignored)
```

## Commands

```bash
source .venv/bin/activate

# main pipeline (modelling/modelling.py)
python Eksperimen_SML_Rudy_Eko/modelling/modelling.py

# tuning pipeline (Workflow-CI) — used by CI
mlflow run Eksperimen_SML_Rudy_Eko/Workflow-CI/MLproject \
  --env-manager=local \
  --experiment-name "KDD Cyber Attack Classify - Tuning"

# view MLflow UI (must run before script — tracking URI is hardcoded)
mlflow ui --port 5000
```

## Key facts

- **MLflow server must be running** before running any script (`mlflow.set_tracking_uri("http://127.0.0.1:5000")` is hardcoded)
- Main `modelling.py` only prints confusion matrix — **no plots or model files are saved** to disk
- `Workflow-CI/MLproject/modelling.py` saves `confusion_matrix_tuning.png` + `best_gnb.pkl` + logs model to MLflow
- Preprocessing: RobustScaler on numeric cols, one-hot encode `protocol_type`, `service`, `flag`; binary outcome
- `.gitignore` excludes: `.venv/`, `__pycache__/`, `mlruns/`, `home/`, `*.log`
- **No tests, lint, or typecheck** config exists in repo

## CI (.github/workflows/ci.yml)

Triggers on push/PR. Pipeline:
1. Install deps from `requirements.txt`
2. `mlflow run` on Workflow-CI MLproject (tuning + saves `best_gnb.pkl`, `confusion_matrix_tuning.png`)
3. Get latest MLflow run_id from `Workflow-CI/mlruns/`
4. Build Docker image (REST API serving port 8080) via `mlflow models build-docker`
5. Push image to Docker Hub (`$DOCKER_USERNAME/kdd-cyber-attack-classify:${{ github.sha }}`)
6. Create GitHub Release with `best_gnb.pkl` + `confusion_matrix_tuning.png`

**Prerequisites:** `DOCKER_USERNAME` and `DOCKER_PASSWORD` must be set as GitHub Secrets.

### Run Docker image locally
```bash
docker run -p 8080:8080 $DOCKER_USERNAME/kdd-cyber-attack-classify:$SHA
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[...feature_values...]]}'
```

## Commit style

`[ADD]` / `[FIX]` / `[EDIT]` prefix in Indonesian or English. No conventional commits.
