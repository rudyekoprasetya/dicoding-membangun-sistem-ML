# AGENTS.md — dicoding-membangun-sistem-ML

**NSL-KDD** intrusion detection (binary: normal→0, attack→1).  
**Model:** GaussianNB — *not* DecisionTreeClassifier.  
**Branch:** `Main` (only branch).

## Pipeline

All under `Eksperimen_SML_Rudy_Eko/`:

1. `preprocessing/automate_preprocessing.py` — RobustScaler on numeric cols, one-hot encode `protocol_type`/`service`/`flag`, remap outcome (normal→0, attack→1)
2. `modelling/modelling.py` — default GaussianNB, logs to MLflow, **no files saved** (CM printed only)
3. `modelling/modelling_tuning.py` — RandomizedSearchCV on `var_smoothing` (loguniform 1e-12–1e-1), saves `best_gnb.pkl` + `confusion_matrix_tuning.png`
4. `Workflow-CI/MLproject/modelling.py` — same tuning but designed for `mlflow run` / CI; does **not** call `set_tracking_uri` or `set_experiment`

## Commands

```bash
source .venv/bin/activate

mlflow ui --port 5000                # must be running before local scripts

# main modelling (hardcoded http://127.0.0.1:5000)
python Eksperimen_SML_Rudy_Eko/modelling/modelling.py

# local tuning (also hardcoded to localhost:5000)
python Eksperimen_SML_Rudy_Eko/modelling/modelling_tuning.py

# CI-style tuning (uses MLFLOW_TRACKING_URI env var, no hardcoded URI)
mlflow run Eksperimen_SML_Rudy_Eko/Workflow-CI/MLproject --env-manager=local
```

## MLflow tracking differences

| Script | set_tracking_uri | set_experiment | autolog |
|--------|-----------------|----------------|---------|
| `modelling.py` | `http://127.0.0.1:5000` | `"KDD Cyber Attack Classify"` | ✓ |
| `modelling_tuning.py` | `http://127.0.0.1:5000` | `"KDD Cyber Attack Classify - Tuning"` | ✓ |
| `Workflow-CI/MLproject/modelling.py` | env var only | ❌ (default experiment) | ❌ |

## CI (.github/workflows/ci.yml)

Trigger: push/PR to `Main`. Python 3.13.

1. `pip install -r requirements.txt`
2. `mlflow run MLproject --env-manager=local` (from `Eksperimen_SML_Rudy_Eko/Workflow-CI/`)
3. Get latest run_id: `ls -td ${{ github.workspace }}/mlruns/*/*/ | head -n1 | xargs basename`
4. `mlflow models build-docker --model-uri "runs:/$RUN_ID/best_gnb_model" --name "kdd-classify"`
5. Push → Docker Hub as `$DOCKER_HUB_USERNAME/kdd-classify:latest`

**Secrets:** `DOCKER_HUB_USERNAME` + `DOCKER_HUB_ACCESS_TOKEN`.

**Known gotcha:** step 3 looks for `mlruns/` at the workspace root, but `mlflow run` with `working-directory` can create it elsewhere — there's a history of CI failures around this (`[FIX] ci run id not found`).

### Docker test locally

```bash
docker run -p 8080:8080 $USER/kdd-classify:latest
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[...feature_values...]]}'
```

## Repo facts

- No tests, lint, typecheck, pyproject.toml, Makefile, or Dockerfile
- `.gitignore`: `.venv/`, `__pycache__/`, `mlruns/`, `home/`, `*.log`, `.env`, `.vscode/`, `.idea/`, `.qodo/`
- `opencode.json` does not exist
- Artifact names in CI docker build: `best_gnb_model` (not `best_gnb`)

## Commit style

`[ADD]` / `[FIX]` / `[EDIT]` / `[DEL]` prefix, Indonesian or English. No conventional commits.
