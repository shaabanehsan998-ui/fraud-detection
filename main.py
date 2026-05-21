from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# تحميل النموذج والـ scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
def predict(data: dict):

    features = np.array(data["features"], dtype=float).reshape(1, -1)

    try:
        scaled = scaler.transform(features)
        pred = model.predict(scaled)

        return {
            "prediction": int(pred[0])
        }

    except Exception as e:
        return {
            "error": str(e)
        }
