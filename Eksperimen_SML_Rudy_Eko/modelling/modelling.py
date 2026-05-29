# ============================================================
# FASE 0: Setup environment dan dependencies
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import subprocess, sys, time, requests, os, signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FASE 1: Auto-start MLflow tracking server
# ============================================================
MLFLOW_PORT = 5000
MLFLOW_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
mlflow_process = None

try:
    requests.get(f"{MLFLOW_URI}/health", timeout=2)
    print(f"MLflow server sudah berjalan di {MLFLOW_URI}")
except requests.exceptions.ConnectionError:
    print("Memulai MLflow server...")
    mlflow_process = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", str(MLFLOW_PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{MLFLOW_URI}/", timeout=2)
            print("MLflow server siap.")
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        print("Gagal menjalankan MLflow server. Cek port 5000.")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("KDD Cyber Attack Classify")

# ============================================================
# FASE 2: Load dataset dan splitting (80% train, 20% test)
# ============================================================
df = pd.read_csv('preprocessed_kdd.csv')

X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size: {X_test.shape[0]} rows")

# ============================================================
# FASE 3: Hyperparameter tuning KNN dengan 10-fold CV dan early stopping
# ============================================================
best_accuracy = 0
best_k = 0
no_improve_count = 0
early_stop_patience = 5
results = []

cv = KFold(n_splits=10, shuffle=True, random_state=42)

print("\nTuning KNN dengan cross-validation 10-fold...")
for k in range(1, 31):
    with mlflow.start_run(run_name=f"KNN_k{k}"):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()

        mlflow.log_param("n_neighbors", k)
        mlflow.log_metric("cv_accuracy", mean_acc)
        mlflow.log_metric("cv_std", std_acc)

        results.append({'k': k, 'accuracy': mean_acc})
        print(f"k={k:2d} | CV accuracy: {mean_acc:.6f} (+/- {std_acc:.6f})")

        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_k = k
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping di k={k} (tidak ada peningkatan selama {early_stop_patience} iterasi)")
            break

# ============================================================
# FASE 4: Evaluasi model terbaik pada test set
# ============================================================
print(f"\nBest k: {best_k} dengan CV accuracy: {best_accuracy:.6f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy dengan k={best_k}: {test_acc:.6f}")

# ============================================================
# FASE 5: Logging final model ke MLflow
# ============================================================
with mlflow.start_run(run_name="Best_KNN_Final"):
    mlflow.log_param("n_neighbors", best_k)
    mlflow.log_metric("cv_accuracy", best_accuracy)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.sklearn.log_model(best_knn, "knn_model")

# ============================================================
# FASE 6: Tampilkan hasil tuning
# ============================================================
print("\nHasil tuning:")
print(pd.DataFrame(results).to_string(index=False))

# ============================================================
# FASE 7: Visualisasi performa (TESTING)
# ============================================================
print("\n--- Visualisasi Performa ---")

df_results = pd.DataFrame(results)
plt.figure(figsize=(10, 5))
plt.plot(df_results['k'], df_results['accuracy'], marker='o', linestyle='-', color='royalblue')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.title('Akurasi Cross-Validation vs Nilai k')
plt.xlabel('k (jumlah tetangga)')
plt.ylabel('Rata-rata CV Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('plot_accuracy_vs_k.png')
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Test Set')
plt.savefig('plot_confusion_matrix.png')
plt.show()

print("Visualisasi performa disimpan: plot_accuracy_vs_k.png, plot_confusion_matrix.png")

# ============================================================
# FASE 8: Export requirements.txt via pip freeze
# ============================================================
subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=open("requirements.txt", "w"))
print("requirements.txt generated via pip freeze")

# ============================================================
# FASE 9: Matikan MLflow server jika dijalankan oleh script ini
# ============================================================
if mlflow_process:
    mlflow_process.terminate()
    print("MLflow server dihentikan.")
