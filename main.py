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
        
        scaled = scaler.transform(features)
        
        prediction = model.predict(scaled)
        
        return {
            "prediction": int(np.array(prediction).ravel()[0])
        }

    except Exception as e:

        return {
            "error": str(e)
        }
        }
