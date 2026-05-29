import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("KDD Cyber Attack Classify")

df = pd.read_csv('preprocessed_kdd.csv')

X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test size: {X_test.shape[0]} rows")

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

print(f"\nBest k: {best_k} dengan CV accuracy: {best_accuracy:.6f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy dengan k={best_k}: {test_acc:.6f}")

with mlflow.start_run(run_name="Best_KNN_Final"):
    mlflow.log_param("n_neighbors", best_k)
    mlflow.log_metric("cv_accuracy", best_accuracy)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.sklearn.log_model(best_knn, "knn_model")

print("\nHasil tuning:")
print(pd.DataFrame(results).to_string(index=False))
