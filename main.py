from fastapi import FastAPI
import joblib
import numpy as np

print("NEW MAIN FILE LOADED")

app = FastAPI()

# تحميل النموذج والـ scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
def predict(data: dict):

    try:

        features = np.array(data["features"], dtype=float).reshape(1, -1)

        print("INPUT SHAPE:", features.shape)
        print("MODEL EXPECTS:", model.n_features_in_)

        scaled = scaler.transform(features)

        pred = model.predict(scaled)

        return {
            "prediction": pred.tolist(),
            "input_shape": str(features.shape),
            "model_expected": str(model.n_features_in_)
        }

    except Exception as e:

        return {
            "error": str(e)
        }
        }
