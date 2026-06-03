import os
import time
import threading
import requests
from prometheus_client import start_http_server, Gauge, Histogram, Counter

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow-app:8080")
PORT = int(os.environ.get("PORT", "8000"))
PROBE_INTERVAL = int(os.environ.get("PROBE_INTERVAL", "10"))

model_health = Gauge("model_health", "Model health status (1=healthy, 0=unhealthy)")
prediction_requests = Counter(
    "prediction_requests_total", "Total prediction requests by class", ["class"]
)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
prediction_correct = Gauge(
    "prediction_correct",
    "Whether prediction matches expected ground truth (1=correct, 0=wrong)",
    ["sample_type"],
)
probe_duration = Gauge(
    "probe_duration_seconds", "Total duration of one probe cycle"
)

SAMPLE_NORMAL = [0, 0, 0, 0, 20, 0, 1.6195652173913044, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0851063829787234, -0.375, 0, 0, 0, 0, 0, 0, 0, -0.6069364161849711, -0.1551020408163265, -0.3578947368421052, 0.1428571428571428, 2.833333333333333, 0, 0, 0, 0.05, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

SAMPLE_ATTACK = [0, 0, 0, 0, 19, 0, -0.1594202898550724, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7730496453900709, -0.125, 1, 1, 0, 0, -1.0439560439560438, 1.1666666666666667, 0, 0, -0.1510204081632653, -0.4315789473684211, 0.4285714285714285, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

SAMPLES = [
    (SAMPLE_NORMAL, "normal", 0),
    (SAMPLE_ATTACK, "attack", 1),
]


def do_probe():
    start_time = time.time()

    try:
        r = requests.get(f"{MLFLOW_URL}/ping", timeout=5)
        model_health.set(1 if r.status_code == 200 else 0)
    except requests.RequestException:
        model_health.set(0)

    for features, sample_type, expected in SAMPLES:
        try:
            t0 = time.time()
            r = requests.post(
                f"{MLFLOW_URL}/invocations",
                json={"inputs": [features]},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            latency = time.time() - t0
            prediction_latency.observe(latency)

            if r.status_code == 200:
                preds = r.json().get("predictions", [])
                for p in preds:
                    cls = "attack" if p == 1 else "normal"
                    prediction_requests.labels(cls).inc()
                    is_correct = 1 if p == expected else 0
                    prediction_correct.labels(sample_type=sample_type).set(is_correct)
        except requests.RequestException:
            prediction_latency.observe(10.0)
            prediction_correct.labels(sample_type=sample_type).set(0)

    probe_duration.set(time.time() - start_time)


def probe_loop():
    while True:
        do_probe()
        time.sleep(PROBE_INTERVAL)


if __name__ == "__main__":
    threading.Thread(target=probe_loop, daemon=True).start()
    start_http_server(PORT)
    print(f"Sidecar started on port {PORT}, probing {MLFLOW_URL} every {PROBE_INTERVAL}s")
    while True:
        time.sleep(1)
