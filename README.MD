# Install requirements:
Run in root: pip3 install -r requirements.txt
# How to run MLflow: run in root terminal
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
