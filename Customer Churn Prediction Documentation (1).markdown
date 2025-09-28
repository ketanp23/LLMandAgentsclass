# Customer Churn Prediction System Documentation

## Overview
This document outlines the process for building, deploying, and monitoring a customer churn prediction system using a Random Forest model. The system uses scikit-learn and pandas for training, Flask for API serving, Docker for containerization, Kubernetes for deployment, and Prometheus/Grafana for monitoring.

## Step 1: Training the Random Forest Model

### Objective
Train a Random Forest model to predict customer churn based on historical data.

### Tools
- **Scikit-learn**: For model training and evaluation.
- **Pandas**: For data preprocessing.
- **Joblib**: For model serialization.

### Process
1. **Data Preparation**:
   - Input: CSV file (`customer_data.csv`) with features (e.g., `tenure`, `usage`, `age`, `monthly_charges`, `contract_type`) and target (`churn`: 1 for churned, 0 for not).
   - For demonstration, synthetic data is generated with 1000 samples.
   - Categorical features (`contract_type`) are one-hot encoded using `pd.get_dummies`.

2. **Model Training**:
   - Split data into 80% training and 20% testing sets.
   - Train a Random Forest Classifier (`n_estimators=100`, `random_state=42`).
   - Evaluate using accuracy and classification report.

3. **Model Serialization**:
   - Save the trained model to `churn_model.pkl` using `joblib`.

### Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Generate synthetic data (replace with pd.read_csv('customer_data.csv') for real data)
np.random.seed(42)
n_samples = 1000
data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'usage': np.random.uniform(0, 100, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'monthly_charges': np.random.uniform(20, 150, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Preprocess: Encode categorical features
df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)

# Split features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'churn_model.pkl')
```

## Step 2: Building the Flask API

### Objective
Serve the trained model via a REST API for real-time predictions.

### Tools
- **Flask**: Lightweight web framework for API.
- **Joblib**: To load the model.
- **Pandas**: For input data preprocessing.

### Process
1. Create `app.py` with a `/predict` endpoint accepting JSON input (e.g., `{"tenure": 12, "usage": 50, "age": 30, "monthly_charges": 70, "contract_type": "Month-to-month"}`).
2. Preprocess input: Encode categorical variables and align with training data columns.
3. Return prediction (`churn`: 0 or 1) and churn probability.

### Code
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)
    # Align columns to match training data
    missing_cols = set(['contract_type_One year', 'contract_type_Two year']) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    prediction = model.predict(df)
    prob = model.predict_proba(df)[0][1]
    return jsonify({'churn': int(prediction[0]), 'probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Testing
Run `python app.py` and test with:
```
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"tenure": 12, "usage": 50, "age": 30, "monthly_charges": 70, "contract_type": "Month-to-month"}'
```

## Step 3: Containerizing with Docker

### Objective
Package the Flask app for scalable deployment.

### Tools
- **Docker**: For containerization.

### Process
1. Create a `Dockerfile` to build a lightweight Python image.
2. Include dependencies in `requirements.txt`.
3. Build and test the container locally.

### Files
**`requirements.txt`**:
```
flask
scikit-learn
pandas
joblib
numpy
prometheus-client
```

**`Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Commands
```
docker build -t churn-predictor .
docker run -p 5000:5000 churn-predictor
```

## Step 4: Deploying on Kubernetes

### Objective
Deploy the containerized app on Kubernetes for scalability.

### Tools
- **Kubernetes**: For orchestration.

### Process
1. Create a `deployment.yaml` for a Deployment and Service.
2. Push the Docker image to a registry (e.g., Docker Hub).
3. Apply the configuration to Kubernetes.

### File
**`deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-predictor
  template:
    metadata:
      labels:
        app: churn-predictor
    spec:
      containers:
      - name: churn-predictor
        image: your-docker-repo/churn-predictor:latest
        ports:
        - containerPort: 5000

---

apiVersion: v1
kind: Service
metadata:
  name: churn-predictor-service
spec:
  selector:
    app: churn-predictor
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

### Commands
```
docker push your-docker-repo/churn-predictor:latest
kubectl apply -f deployment.yaml
```

## Step 5: Monitoring with Prometheus and Grafana

### Objective
Monitor prediction latency and potential accuracy drift.

### Tools
- **Prometheus**: For metrics collection.
- **Grafana**: For visualization.
- **Prometheus-client**: For instrumenting Flask.

### Process
1. Add Prometheus metrics to `app.py` for request count and latency.
2. Expose a `/metrics` endpoint.
3. Configure Prometheus to scrape metrics from Kubernetes pods.
4. Set up Grafana dashboards for visualization.

### Updated `app.py` (Monitoring Section)
```python
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from flask import Response

# Metrics
requests_total = Counter('requests_total', 'Total requests')
request_latency = Histogram('request_latency_seconds', 'Request latency')

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
@request_latency.time()
def predict():
    requests_total.inc()
    # ... (existing predict code)
```

### Prometheus Configuration
**`prometheus.yml`**:
```yaml
scrape_configs:
  - job_name: 'churn-predictor'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: churn-predictor
```

### Monitoring Setup
- Deploy Prometheus and Grafana in Kubernetes (use Helm charts for simplicity).
- Create Grafana dashboards for:
  - Request latency (`request_latency_seconds` histogram).
  - Request volume (`requests_total` counter).
  - Accuracy drift (requires logging predictions and ground truth to a database; use tools like Alibi Detect for drift detection).

### Notes on Accuracy Drift
- Log predictions and actual outcomes to a database (e.g., PostgreSQL).
- Periodically compare predictions to ground truth to detect drift.
- Retrain the model if drift is detected (use a pipeline like Airflow or Kubeflow).

## Key Insights
- **Lightweight Models**: Random Forest models are computationally efficient, enabling easy deployment without GPUs.
- **Robust Pipelines**: Ensure data pipelines handle feature engineering, validation, and retraining to maintain model performance.
- **Monitoring**: Latency and request metrics are critical for production; accuracy drift monitoring requires a separate pipeline for ground truth collection.

## Prerequisites
- Python 3.12, Docker, Kubernetes cluster, Prometheus, Grafana.
- Docker registry access for image storage.
- Optional: Airflow/Kubeflow for pipeline orchestration, Alibi Detect for drift detection.

## Maintenance
- Regularly update the model with new data.
- Monitor Prometheus alerts for latency spikes or errors.
- Retrain if accuracy drift exceeds a threshold (e.g., 10% drop in precision).