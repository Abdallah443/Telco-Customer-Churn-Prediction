import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_DIR = BASE_DIR / "model"
DB_PATH = BASE_DIR / "data" / "customer_records.db"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

with open(MODEL_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
    FEATURE_META = json.load(f)

FEATURES = FEATURE_META["numeric"] + FEATURE_META["categorical"]

with open(MODEL_DIR / "churn_model.pkl", "rb") as f:
    MODEL = pickle.load(f)


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(payload, prediction, probability):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO customer_predictions (created_at, payload, prediction, probability)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(payload, ensure_ascii=False),
                prediction,
                float(probability),
            ),
        )
        conn.commit()


def normalize_field(key, value):
    if key in ["SeniorCitizen", "tenure"]:
        if isinstance(value, str):
            clean = value.strip()
            if clean.lower() in ["yes", "y", "true"]:
                return 1 if key == "SeniorCitizen" else int(clean)
            if clean.lower() in ["no", "n", "false"]:
                return 0 if key == "SeniorCitizen" else int(clean)
            return int(float(clean))
        return int(float(value))

    if key in ["MonthlyCharges", "TotalCharges"]:
        return float(value)

    if value is None:
        return ""
    return str(value).strip()


def build_feature_row(raw_payload):
    errors = []
    payload = {}

    for key in FEATURES:
        if key not in raw_payload:
            errors.append(f"Missing field: {key}")
            continue
        payload[key] = normalize_field(key, raw_payload[key])

    if errors:
        raise ValueError("; ".join(errors))

    return pd.DataFrame([payload], columns=FEATURES)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        raw_payload = request.get_json(silent=True) or {}
        row = build_feature_row(raw_payload)
        prediction = MODEL.predict(row)[0]
        probabilities = MODEL.predict_proba(row)[0]
        label = "Yes" if int(prediction) == 1 else "No"
        confidence = float(probabilities[int(prediction)])

        payload = row.iloc[0].to_dict()
        save_prediction(payload, label, confidence)

        return jsonify(
            {
                "prediction": label,
                "probability": round(confidence, 4),
                "confidence": round(confidence * 100, 2),
                "features": payload,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, payload, prediction, probability
            FROM customer_predictions
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()

    return jsonify([
        {
            "created_at": dict(row)["created_at"],
            "payload": json.loads(dict(row)["payload"]),
            "prediction": dict(row)["prediction"],
            "probability": dict(row)["probability"],
        }
        for row in rows
    ])


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


ensure_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
