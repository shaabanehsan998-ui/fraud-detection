from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "NEW VERSION WORKING"
    }

@app.post("/predict")
def predict():
    return {
        "status": "API UPDATED SUCCESSFULLY"
    }
