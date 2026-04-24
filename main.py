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

<<<<<<< HEAD
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
=======
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_model, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_model}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Find the best model based on F1 score
    best_model_name = max(all_metrics, key=lambda k: sum(all_metrics[k][m] for m in ["F1", "Accuracy"]))
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )
    logging.info("Final evaluation completed.")

if __name__ == "__main__":
    main()
>>>>>>> 4c0272762fd503b86a8a69d49046228cb2f5c054
