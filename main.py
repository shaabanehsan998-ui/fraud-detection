from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

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

        features = np.array(data["features"], dtype=float)

        features = features.reshape(1, -1)

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:

        return {
            "error": str(e)
        }
