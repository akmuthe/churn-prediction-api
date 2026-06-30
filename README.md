# Customer Churn Prediction API

A containerized, deployed machine learning API that predicts customer churn probability in real time, with built-in request logging for monitoring.

## What this project demonstrates
- Training and evaluating a classification model (Random Forest, scikit-learn) — ROC-AUC ~0.88
- Serving the model as a REST API (Flask + Gunicorn)
- Containerizing the application with Docker
- Deploying to a cloud platform (Render free tier)
- Basic production monitoring: every prediction request is logged with timestamp, inputs, output, and latency; a `/stats` endpoint summarizes logged predictions

## Project structure
```
churn-api/
├── train_model.py       # generates data, trains and evaluates the model
├── app.py                # Flask API: /predict, /health, /stats
├── requirements.txt
├── Dockerfile
├── churn_model.pkl        # trained model artifact
├── encoders.pkl           # label encoders for categorical features
├── feature_cols.pkl
└── prediction_log.csv     # auto-generated log of all predictions
```

## API endpoints

### `GET /health`
Returns service status.

### `POST /predict`
Request body:
```json
{
  "tenure": 2,
  "monthly_charges": 95.5,
  "total_charges": 191.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "payment_method": "Electronic check"
}
```
Response:
```json
{
  "churn_prediction": 1,
  "churn_label": "Likely to churn",
  "churn_probability": 0.9836,
  "latency_ms": 28.86
}
```

### `GET /stats`
Returns summary statistics from logged predictions (total count, predicted churn rate, average latency) — a lightweight monitoring view.

## Running locally
```bash
pip install -r requirements.txt
python train_model.py    # trains model and saves artifacts (run once)
python app.py             # starts the API on localhost:5050
```

## Running with Docker
```bash
docker build -t churn-api .
docker run -p 5050:5050 churn-api
```

## Deployment
Deployed on Render's free tier as a Docker web service. See `DEPLOY.md` for step-by-step instructions.

## Notes
The dataset used here is synthetically generated (see `train_model.py`) for demonstration purposes, with churn probability driven by realistic features: tenure, contract type, internet service, tech support, and payment method.
