from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# تحميل النموذج والـ scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }


@app.post("/predict")
def predict(data: dict):

    try:

        # أخذ البيانات
        features = data["features"]

        # تحويلها لمصفوفة
        features = np.array(features, dtype=float)

        # تحويلها لشكل مناسب
        features = features.reshape(1, -1)

        # scaling
        features_scaled = scaler.transform(features)

        # prediction
        prediction = model.predict(features_scaled)

        # probability
        probability = model.predict_proba(features_scaled)

        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(probability[0][1])
        }

    except Exception as e:

        return {
            "prediction": int(prediction[0]),
            "probability": str(probability)
        }
