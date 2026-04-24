from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
from typing import Literal
import pandas as pd

# ====== Logging setup ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== App init ======
app = FastAPI(
    title="Lung Cancer Prediction API",
    version="1.0.0"
)

# ====== Input schema ======
class LungCancerInput(BaseModel):
    age: float
    weight_change: float
    smoking_duration: float
    gender: Literal[0, 1]
    gene_markers: float
    air_pollution_level: Literal[0, 1, 2]
    tiredness_frequency: Literal[0, 1, 2]

# ====== Lazy model Loading ======
model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading model...")
        model = joblib.load("src/artifacts/lung_cancer_model.pkl")
    return model

# ====== Health check ======
@app.get("/health")
def health():
    return{"status": "ok"}

# ====== Prediction endpoint ======
@app.post("/predict")
def predict(data: LungCancerInput):
    model = load_model()
    input_dict = data.model_dump()
    feature_columns = [
        "age",
        "weight_change",
        "smoking_duration",
        "gender",
        "gene_markers",
        "air_pollution_level",
        "tiredness_frequency"
    ]
    df = pd.DataFrame([input_dict])[feature_columns]
        
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }