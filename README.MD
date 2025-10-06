# 🏠 Irish Housing Market Drift Monitor

An end-to-end **MLOps project** that predicts and monitors housing price trends in Ireland using real data from the CSO (Central Statistics Office).  
The project tracks **data drift**, retrains the model automatically when drift exceeds a threshold, and logs all experiments to **MLflow** for full experiment tracking and model versioning.

---

## ⚙️ Features

✅ Predicts mean–median price gap ratios across Irish counties (2010–2023).  
✅ Tracks model drift over time using **Evidently AI**.  
✅ Automatically logs metrics and models to **MLflow**.  
✅ Streams drift metrics to **Prometheus** and visualizes them in **Grafana**.  
✅ Hourly drift monitoring and retraining managed by **Apache Airflow**.  

---

## 🧩 Tech Stack

| Component | Purpose |
|------------|----------|
| **Python (scikit-learn, pandas)** | Data preprocessing & model training |
| **MLflow** | Experiment tracking, model registry |
| **Evidently AI** | Data drift detection & HTML drift reports |
| **Prometheus** | Metrics scraping from Evidently service |
| **Grafana** | Drift visualization dashboards and alerts |
| **Airflow** | Scheduled drift checks and automated retraining |
| **CSO JSONStat API** | Source for Irish housing data |

---

## 🚀 Setup Instructions

### 1️⃣ Clone and enter the repo
```bash
git clone https://github.com/<your-username>/irish-house-model.git
cd irish-house-model
```
### 2. Create and activate the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies/python dependencies
``` bash
brew install prometheus
pip3 install -r requirements.txt
```
### 4. Run services
``` bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000

prometheus --config.file=prometheus.yml

brew services start grafana
```

## Note: If running on Mac0S, export these variables in terminal
``` bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export no_proxy="*"
export PYTHONFAULTHANDLER=1
```

## Typical Workflow
1. Typical Workflow

2. Train your regression model to predict the gap ratio between mean and median prices.

3. Log the model and metrics to MLflow.

4. Generate a drift report comparing historical vs. new data.

5. Expose drift metrics to Prometheus.

6. Visualize trends and configure alerts in Grafana.

## Future Improvements
* Future Improvements

* Automated retraining when drift exceeds a threshold.

* Web dashboard for first-time home buyers to explore data by county.

* Slack/email integration for automated drift alerts.
