# ============================================================
# FASE 0: Setup environment dan dependencies
# ============================================================
import numpy as np
import pandas as pd

import hashlib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import subprocess, sys, os
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

mlflow.set_experiment("KDD Cyber Attack Classify")
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

# Log parameter dataset
with mlflow.start_run():
    mlflow.log_param("n_samples_total", n_samples_total)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("train_size", train_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_artifact(csv_path)

    # Log input dataset via mlflow.data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_dataset = mlflow.data.from_pandas(train_df, source=csv_path, name="kdd_train", targets="outcome")
    test_dataset = mlflow.data.from_pandas(test_df, source=csv_path, name="kdd_test", targets="outcome")
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="test")

# ============================================================
# FASE 3: Training GaussianNB + Evaluasi
# ============================================================
print("\nMelatih GaussianNB...")

with mlflow.start_run(run_name="NaiveBayes_Model"):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.6f}")

    mlflow.log_metric("test_accuracy", test_acc)

    # Confusion Matrix — print only
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    print(cm)

# ============================================================
# FASE 4: Selesai
# ============================================================
print("\nSemua fase selesai. buka http://127.0.0.1:5000 untuk melihat tracking ML Training")

