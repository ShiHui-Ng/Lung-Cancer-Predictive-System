# Use Gradient Boosting, Random Forest and Decision Trees

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Algorithm imports
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Data splitting and evaluation metric imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("./data/data_clean.csv")
print(df.head())
print("")

# Separate data into features and target
X = df.drop(columns=['lung_cancer'])
y = df['lung_cancer']

# Split data into training (80%) and test-validation (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2 , random_state=42)

# Split test-validation set into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Categorise features
numerical_features = ['age', 'weight_change', 'smoking_duration']
nominal_features = ['gender', 'gene_markers']
ordinal_features = ['air_pollution_level', 'tiredness_frequency']
passthrough_features = []

# Define categories for ordinal features
ordinal_categories = [
    ['Low', 'Medium', 'High'], # Categories for 'air_pollution_level'
    ['Low', 'Medium', 'High']  # Categories for 'tiredness_frequency'
]

# Preprocess data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='error'))
])
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='error'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('nom', nominal_transformer, nominal_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('pass', 'passthrough', passthrough_features)
    ],
    remainder='passthrough',
    n_jobs=-1
    )

# Create machine learning pipelines

dt_pipeline = Pipeline(steps=[ # Decision Tree Classifier pipeline
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeClassifier(random_state=42))
])

cb_pipeline = Pipeline(steps=[ # Catboost classifier pipeline 
    ('preprocessor', preprocessor),
    ('catboost', CatBoostClassifier(random_state=42, verbose=0))
])

xgb_pipeline = Pipeline(steps=[ # XGBoost Classifier pipeline
    ('preprocessor',preprocessor),
    ('xgboost', XGBClassifier(random_state=42))
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])
def model_training(model, title1, title2):
    '''
    Train machine learning baseline model used for comparison against other models' performances.

    Parameters:
    -----------
    model_val (Pipeline):           A machine learning pipeline fitted to validation dataset.
    y_val_pred (pd.Series):         A prediction on validation set.
    y_val_pred_proba (pd.Series):   A prediction on probability of class 1 occurring in each category.
    model_train (Pipeline):         A variable storing the trained pipeline.
    y_train_pred (pd.Series):       A prediction on training set.
    y_train_pred_proba (pd.Series): A prediction on probability of class 1 occurring in each category.

    Returns:
    Evaluation metrics of baseline model.
    '''
    # Fit dt pipeline to training data
    model = model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate evaluation metrics for validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    # Predict on training set
    y_train_pred = dt_pipeline.predict(X_train)
    y_train_pred_proba = dt_pipeline.predict_proba(X_train)[:, 1]
    
    # Calculate evaluation metrics for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

    print(title1)
    print(f'Validation Accuracy: {val_accuracy:.5f}')
    print(f'Validation Precision: {val_precision:.5f}')
    print(f'Validation Recall: {val_recall:.5f}')
    print(f'Validation F1 Score: {val_f1:.5f}')
    print(f'Validation ROC AUC: {val_roc_auc:.5f}')
    print("")
    print("")
    print(title2)
    print(f'Training Accuracy: {train_accuracy:.5f}')
    print(f'Training Precision: {train_precision:.5f}')
    print(f'Training Recall: {train_recall:.5f}')
    print(f'Training F1 Score: {train_f1:.5f}')
    print(f'Training ROC AUC: {train_roc_auc:.5f}')

# Decision Tree Model
model_training(dt_pipeline, "Decision Tree Validation Set Metrics:", "Decision Tree Training Set Metrics") 
print("")

# Catboost Model
model_training(cb_pipeline, "Catboost Validation Set Metrics:", "Catboost Training Set Metrics")
print("")

# XGBoost Model
model_training(xgb_pipeline, "XGBoost Validation Set Metrics", "XGBoost Training Set Metrics")
print("")

# Random Forest Model
model_training(rf_pipeline, "Random Forest Validation Set Metrics", "Random Forest Training Set Metrics")
'''
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('catboost', CatBoostClassifier(
        iterations=1000, 
        depth=6, 
        learning_rate=0.01, 
        loss_function='Logloss', 
        eval_metric='F1',
        l2_leaf_reg=6,
        verbose=0, 
        random_state=42))
])

param_grid = {
    'catboost__iterations':            [100, 1000],
    'catboost__depth':                 [4, 16],
    'catboost__learning_rate':         [0.01, 0.3],
    'catboost__loss_function':         ['Logloss','RMSE'],
    'catboost__l2_leaf_reg':           [1, 10]
}

grid_search =  GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
'''
'''
depth: 4,
iterations: 100
leaf_reg: 10
learning_rate: 0.01
loss_function: logloss

Best Accuracy: 0.7802173432621697
Test Accuracy: 0.7029063509149623
F1 Score: 0.764102564102564
Precision: 0.6468885672937771
Recall: 0.9331941544885177
ROC AUC Score: 0.8136116910229645
'''
