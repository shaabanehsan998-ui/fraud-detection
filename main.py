
from fastapi import FastAPI
import joblib
import numpy as np

# إنشاء التطبيق
app = FastAPI()

# تحميل النموذج
model = joblib.load("model.pkl")

# تحميل scaler
scaler = joblib.load("scaler.pkl")

# الصفحة الرئيسية
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API Running"
    }

# endpoint للتنبؤ
@app.post("/predict")
def predict(data: dict):

    # تحويل البيانات إلى numpy array
    features = np.array(data["features"]).reshape(1, -1)

    # scaling
    features_scaled = scaler.transform(features)

    # prediction
    prediction = model.predict(features_scaled)

    return {
        "prediction": int(prediction[0])
    }
