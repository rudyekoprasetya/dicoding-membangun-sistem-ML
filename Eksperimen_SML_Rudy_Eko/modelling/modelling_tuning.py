# ============================================================
# FASE 0: Setup environment dan dependencies
# ============================================================
import numpy as np
import pandas as pd

import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import loguniform
import mlflow
import mlflow.sklearn
import subprocess, sys, os, pickle
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# FASE 1: Setup MLflow tracking (remote server)
# ============================================================
# Pastikan MLflow server sudah jalan: mlflow ui --port 5000
script_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()
csv_path = os.path.join(script_dir, 'preprocessed_kdd.csv')

# ============================================================
# FASE 1: Dataset tracking — hash, parameter, dan artifact
# ============================================================
with open(csv_path, 'rb') as f:
    dataset_hash = hashlib.md5(f.read()).hexdigest()

mlflow.set_experiment("KDD Cyber Attack Classify - Tuning")
with mlflow.start_run(run_name="Dataset_Info"):
    mlflow.log_param("dataset_source", os.path.basename(csv_path))
    mlflow.log_param("dataset_hash", dataset_hash)

# ============================================================
# FASE 2: Load dataset dan splitting (80% train, 20% test)
# ============================================================
df = pd.read_csv(csv_path)

X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_samples_total = len(df)
n_features = X.shape[1]
train_size = X_train.shape[0]
test_size = X_test.shape[0]

print(f"Total samples: {n_samples_total}")
print(f"Train size: {train_size} rows")
print(f"Test size: {test_size} rows")

with mlflow.start_run():
    mlflow.log_param("n_samples_total", n_samples_total)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("train_size", train_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_artifact(csv_path)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_dataset = mlflow.data.from_pandas(train_df, source=csv_path, name="kdd_train", targets="outcome")
    test_dataset = mlflow.data.from_pandas(test_df, source=csv_path, name="kdd_test", targets="outcome")
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="test")

# ============================================================
# FASE 3: RandomizedSearchCV — GaussianNB
# ============================================================
print("\nRandomizedSearchCV — GaussianNB...")

param_dist = {'var_smoothing': loguniform(1e-12, 1e-1)}
n_iter = 20
cv = 5

gnb = GaussianNB()
random_search = RandomizedSearchCV(
    estimator=gnb,
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=cv,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

with mlflow.start_run(run_name="RandomizedSearch_GNB"):
    mlflow.log_param("n_iter", n_iter)
    mlflow.log_param("cv_folds", cv)
    mlflow.log_param("scoring", "accuracy")
    mlflow.log_param("search_type", "RandomizedSearchCV")
    mlflow.log_param("param_distribution", str(param_dist))

    random_search.fit(X_train, y_train)

    best_gnb = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {best_cv_score:.6f}")

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_accuracy", best_cv_score)

    for i in range(len(random_search.cv_results_['params'])):
        params = random_search.cv_results_['params'][i]
        mean_score = random_search.cv_results_['mean_test_score'][i]
        with mlflow.start_run(run_name=f"GNB_iter_{i+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mean_test_score", mean_score)

# ============================================================
# FASE 4: Evaluasi best model di test set
# ============================================================
print("\nEvaluasi best model di test set...")

y_pred = best_gnb.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.6f}")

cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

report = classification_report(y_test, y_pred, target_names=['normal', 'attack'])
print("\n--- Classification Report ---")
print(report)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
plt.title('Confusion Matrix - Best GaussianNB')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(script_dir, 'confusion_matrix_tuning.png')
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

model_path = os.path.join(script_dir, 'best_gnb.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_gnb, f)

with mlflow.start_run(run_name="Best_Model_Evaluation"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("best_cv_accuracy", best_cv_score)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(best_gnb, "best_gnb_model")

# ============================================================
# FASE 5: Selesai
# ============================================================
print("\nSemua fase selesai. Buka http://127.0.0.1:5000 untuk melihat tracking ML Training")
