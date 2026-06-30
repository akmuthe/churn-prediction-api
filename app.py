import time
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

LOG_FILE = "prediction_log.csv"

# Write CSV header once if file doesn't exist yet
import os
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,tenure,monthly_charges,total_charges,contract,internet_service,"
                "tech_support,payment_method,prediction,churn_probability,latency_ms\n")


def log_prediction(line):
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---- Load model artifacts ----
model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": {
            "GET /health": "service health check",
            "POST /predict": "predict churn for a customer (see README for request format)",
            "GET /stats": "summary stats of predictions made so far"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        payload = request.get_json()

        required = ["tenure", "monthly_charges", "total_charges", "contract",
                    "internet_service", "tech_support", "payment_method"]
        missing = [f for f in required if f not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = {}
        for col in ["contract", "internet_service", "tech_support", "payment_method"]:
            le = encoders[col]
            if payload[col] not in le.classes_:
                return jsonify({"error": f"Invalid value for {col}: {payload[col]}. "
                                          f"Expected one of {list(le.classes_)}"}), 400
            row[col] = le.transform([payload[col]])[0]

        row["tenure"] = payload["tenure"]
        row["monthly_charges"] = payload["monthly_charges"]
        row["total_charges"] = payload["total_charges"]

        X = pd.DataFrame([row])[feature_cols]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        latency_ms = round((time.time() - start) * 1000, 2)

        # log every request for monitoring/auditing
        log_line = (f"{datetime.utcnow().isoformat()},{payload['tenure']},"
                    f"{payload['monthly_charges']},{payload['total_charges']},"
                    f"{payload['contract']},{payload['internet_service']},"
                    f"{payload['tech_support']},{payload['payment_method']},"
                    f"{pred},{prob:.4f},{latency_ms}")
        log_prediction(log_line)

        return jsonify({
            "churn_prediction": pred,
            "churn_label": "Likely to churn" if pred == 1 else "Likely to stay",
            "churn_probability": round(prob, 4),
            "latency_ms": latency_ms,
        })

    except Exception as e:
        log_prediction(f"{datetime.utcnow().isoformat()},ERROR,{str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Simple monitoring endpoint: summarizes logged predictions so far."""
    try:
        df = pd.read_csv("prediction_log.csv")
        df = df[df["prediction"].notna()]
        if len(df) == 0:
            return jsonify({"total_predictions": 0})
        return jsonify({
            "total_predictions": len(df),
            "churn_rate_predicted": round(df["prediction"].astype(float).mean(), 3),
            "avg_latency_ms": round(df["latency_ms"].astype(float).mean(), 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)