# Plan: Tambah Docker build + push + GitHub Release ke CI workflow

## File yang diubah
- `.github/workflows/ci.yml`

## Perubahan

### 1. Fix path `Get latest MLflow run_id`
**Before:**
```yaml
RUN_ID=$(ls -t Eksperimen_SML_Rudy_Eko/Workflow-CI/MLproject/mlruns/*/*/meta.yaml ...)
```
**After:**
```yaml
RUN_ID=$(ls -t Eksperimen_SML_Rudy_Eko/Workflow-CI/mlruns/*/*/meta.yaml ...)
```
Karena `mlflow run` membuat `mlruns/` di working-directory (`Workflow-CI/`), bukan di dalam `MLproject/`.

### 2. Build Docker image (serving REST API via `mlflow models build-docker`)
```yaml
- name: Build Docker image serving model (REST API)
  if: env.MLFLOW_RUN_ID != ''
  env:
    MLFLOW_TRACKING_URI: ${{ github.workspace }}/Eksperimen_SML_Rudy_Eko/Workflow-CI/mlruns
  run: |
    IMAGE="${{ secrets.DOCKER_USERNAME }}/kdd-cyber-attack-classify:${{ github.sha }}"
    mlflow models build-docker \
      --model-uri "runs:/${{ env.MLFLOW_RUN_ID }}/best_gnb_model" \
      --name "$IMAGE"
    echo "IMAGE_NAME=$IMAGE" >> $GITHUB_ENV
```
Image serve model via REST API di port 8080 (default MLflow).

### 3. Push Docker image ke Docker Hub
```yaml
- name: Push Docker image to Docker Hub
  if: env.IMAGE_NAME != ''
  run: |
    echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
    docker push "${{ env.IMAGE_NAME }}"
```

### 4. GitHub Release dengan artifacts model
```yaml
- name: Create GitHub Release with artifacts
  if: env.MLFLOW_RUN_ID != ''
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    TAG_NAME="model-build-${{ github.sha }}"
    RELEASE_NOTES="..."
    gh release create "$TAG_NAME" \
      --title "Model Build ${{ github.sha }}" \
      --notes "$RELEASE_NOTES" \
      ./Eksperimen_SML_Rudy_Eko/Workflow-CI/MLproject/best_gnb.pkl \
      ./Eksperimen_SML_Rudy_Eko/Workflow-CI/MLproject/confusion_matrix_tuning.png
```

## Prasyarat (GitHub Secrets)
| Secret | Deskripsi |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token / password |

## Cara pakai Docker image
```bash
docker run -p 8080:8080 $DOCKER_USERNAME/kdd-cyber-attack-classify:$GIT_SHA
# POST prediction:
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [...]}'
```

Setelah deploy, update `AGENTS.md` dengan informasi workflow baru.
