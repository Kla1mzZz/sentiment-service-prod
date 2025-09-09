# Sentiment Service

A microservice for sentiment analysis of text using a machine learning model, FastAPI, and Docker.  

The service accepts text input, returns a sentiment prediction (positive/negative).

---

## 🚀 Technologies

- **FastAPI** — building REST API.  
- **scikit-learn** — machine learning model for sentiment analysis.  
- **Docker** — containerization for easy deployment.  
- **MLflow** (optional) — tracking predictions and model versions.  
- **JWT** — authentication for secure API access.  

---

## 📂Project Structure

```bash
│   Dockerfile
│   poetry.lock
│   pyproject.toml
│   README.md
│
├───data
│       dataset.csv
│
├───models
│       sentiment_pipeline.joblib
│
├───scripts
│       train.py
│
├───src
│   └───sentiment_service
│       │   app.py
│       │   __init__.py
│       │
│       ├───routers
│       │   └─── predict.py
│       │
│       ├───schemas
│       │   └─── input.py
│
└───tests
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kla1mzZz/sentiment-service-prod.git
cd sentiment-service-prod
```

### 2. Run in docker

```bash
docker build -t sentiment-service .
docker build -t sentiment-service .
```

