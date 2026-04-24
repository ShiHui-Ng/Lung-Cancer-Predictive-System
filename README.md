<<<<<<< HEAD
# <img src="https://img.freepik.com/free-vector/human-lungs-with-face-expression_1308-126846.jpg" width="50"> Lung Cancer Risk Prediction System.
## <img src="https://img.freepik.com/premium-vector/cartoon-magnifying-glass-icon-vector-illustration_1138841-30500.jpg" width="50"> Overview
### This project is a machine learning system designed to predict lung cancer risk based on clinical and lifestyle-related features.
##
### It includes:
* End-to-end ML pipeline (data preprocessing -> model training -> evaluation)
* Multiple baseline and tuned models
* Automated model selection
* REST API deployment using FastAPI
##
#### <img src="https://cdn.pixabay.com/animation/2023/04/28/18/34/18-34-10-554_512.gif" width="50"> Disclaimer: This is a mini project used to serve as a prototype for various machine learning models and should not be used to fully define the risks of lung cancer.
##
## <img src="https://thumbs.dreamstime.com/b/building-construction-cartoon-vector-illustration-d-isometric-concept-house-77161070.jpg" width="50"> System Architecture
'''
A[DB Data Source] --> B[Data Loading Layer]
B --> C[DataPrep: Cleaning and Feature Engineering]
C --> D[Train / Validate / Test Split]

D --> E1[Baseline Models]
D --> E2[CatBoost Hyperparameter Tuning]

E1 --> F[Model Evaluation]
E2 --> F

F --> G[Best Model Selection (F1 + Accuracy)]

G --> H[Save Model with Joblib]

H --> I[FastAPI Inference API]
I --> J[Prediction Endpoint /predict]
'''
## <img src="https://www.shutterstock.com/shutterstock/videos/27735403/thumb/1.jpg" width="50> Tech Stack
* Python 3.11
* Scikit-learn
* CatBoost / XGBoost / Random Forest
* Pandas
* FastAPI
* Joblib
* PostgreSQL(planned / optional upgrade)
##
## <img src="https://img.freepik.com/free-vector/science-molecule-atom-icon-design_24640-134135.jpg" width="50"> Features
### Data Pipeline
* Automated preprocessing pipeline using ColumnTransformer
* Categorical encoding (OneHot + Ordinal)
* Feature engineering (smoking duration, weight change, etc.)
### Machine Learning
* Multiple baseline models:
   * Decision Tree
   * Random Forest
   * XGBoost
   * CatBoost
* Hyperparameter tuning using GridSearchCV
* Model selection based on weighted F1 + Accuracy score
### Deployment
* REST API using FastAPI
* Real-time prediction endpoint
* Model persistence using Joblib
##
## <img src="https://thumb.r2.moele.me/t/31528/31518027/a-0120.jpg" width="50> Model Performance
##
### Baseline Models (Validation Set)
#
|       Model          |    Accuracy    |    Precision    |     Recall     |        F1        |       ROC AUC      |
|----------------------|----------------|-----------------|----------------|------------------|--------------------|
| Decision Tree        | 0.677072       | 0.681274        | 0.709543       | 0.695121         | 0.676808           |
| Catboost             | 0.717976       | 0.693661        | 0.817427       | 0.750476         | 0.824667           |
| XGBoost              | 0.728740       | 0.710622        | 0.804979       | 0.754863         | 0.809924           |
| Random Forest        | 0.712594       | 0.700934        | 0.778008       | 0.737463         | 0.800627           |
|------------------------------------------------------------------------------------------------------------------|
| Selected Model: XGBoost (Best balance of F1 + Accuracy score)                                                    |
|------------------------------------------------------------------------------------------------------------------|

## Final Test Performance
|       Model          |    Accuracy    |       F1        |     ROC AUC     |
|---------------------------------------------------------------------------|
| XGBoost              | 0.727664       | 0.719008        | 0.819362        |

## <img src="https://thumb.r2.moele.me/t/31528/31518027/a-0120.jpg" width="50"> Key Observations
* Dataset is primarily categorical in nature
* Moderate signal strength between features and target
* Weight change is one of the most influential features
* Genetic markers show strong predictive signal
* Age distribution is skewed towards older population (60+)
### <img src="https://cdn.pixabay.com/animation/2023/04/28/18/34/18-34-10-554_512.gif" width="50> Note: Observations are dataset-specific and not medical conclusions

## <img src="https://media3.giphy.com/media/v1.Y2lkPTZjMDliOTUybjViM3MxYzQ5aG53bmE1MWdmazZiem4yczBvYXBuNjJ6dnVqMGg2ZCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Swa7CpzF7xTwwK6VWQ/source.gif" width="50"> Feature Engineering
##
|         Feature           |                           Data handling Description                             | 
|---------------------------|---------------------------------------------------------------------------------|
| Age                       | Absolute value transformation                                                   |
| Weight change             | Derived from current and previous weight                                        |
| Missing categorical values| Dropped to ensure data integrity and reduce noise in model training             |
| Smoking duration          | Derived from start and stop smoking times                                       |
| Categorical Features      | Standardized and encoded                                                        |
| ID field                  | Removed due to irrelevancy                                                      |
|-------------------------------------------------------------------------------------------------------------|

## <img src="https://img.freepik.com/free-vector/rocket-flying-space-cartoon-vector-icon-illustration-science-technology-icon-concept-isolated_138676-7558.jpg" width="50"> API Deployment (FastAPI)
### Endpoint
* Post /predict

## <img src="https://www.shutterstock.com/shutterstock/videos/32003665/thumb/9.jpg" width="50"> How to run
1. pip install -r requirements.txt (Install dependencies)
2. python src/run_pipeline.py (Train model)
3. uvicorn main:app --reload (Start API)

## Design Decisions
* XGBoost is selected due to its strong performance on categorical-heavy datasets
* Pipeline architecture is used to ensure reproducibility
* Weighted model selection prioritizes F1 over accuracy due to class imbalance
* Modular design separates:
    * data preprocessing
    * training
    * inference

## <img src="https://cdn.pixabay.com/animation/2023/04/28/18/34/18-34-10-554_512.gif" width="50"> Limitations
* No real-time data drift monitoring
* No CI/CD pipeline yet
* Model retraining is manual
* No production database logging (planned PostgreSQL integration)

## <img src="https://i.pinimg.com/originals/ab/f3/6d/abf36d14c63c1111fabda98ee1071877.gif" width="50"> Future Improvements
* Add PostgreSQL logging for predictions
* Dockerize deployment
* Add MLflow tracking
* Implement model monitoring (drift detection)
* Deploy on cloud (AWS/GCP/Azure)
