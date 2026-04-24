import logging
import os
from typing import Any, Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
import joblib
import pandas as pd
import yaml

from data_prep import DataPrep

logging.basicConfig(level=logging.INFO)

class ModelTraining:
    def __init__(self, config: Dict[str, Any], preprocesor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Parameter:
        -----
        - config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        - preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocesor

    def split_data(
            self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Parameter:
        -----
        - df (pd.DataFrame): The input DataFrame containing the cleaned data.
        
        Returns:
        --------
        Tuple (DataFrame, Series): A tuple containing the training, validation, test features and target variables.
        """
        logging.info("Starting data splitting.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config["val_test_size"] , random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42
        )
        logging.info("Data splitted into training(70%), validation(10%) and testing(20%) sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_and_evaluate_baseline_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train and evaluate baseline models.

        Parameters:
        -----------
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline classifier models.")
        models = {
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "catboost": CatBoostClassifier(verbose=0, random_state=42),
            "xgboost": XGBClassifier(random_state=42),
            "random_forest": RandomForestClassifier(random_state=42)
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self.evaluate_validation(
                pipeline, X_val, y_val, model_name
            )
        return pipelines, metrics
    
    def train_and_evaluate_tuned_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for Catboost model.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val(pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_model = {}
        tuned_metrics = {}
        param_grid = self.config["param_grid"]
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        model = {'catboost_tuned': CatBoostClassifier(verbose=0, random_state=42)}

        for model_name, model in model.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            )
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            tuned_model[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self.evaluate_validation(
                tuned_model[model_name], X_val, y_val, model_name + " (tuned)"
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_model, tuned_metrics
    
    def evaluate_test_set(
            self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Parameters:
        -----------
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
             "Accuracy": accuracy_score(y_test, y_test_pred),
             "Precision": precision_score(y_test, y_test_pred),
             "Recall": recall_score(y_test, y_test_pred),
             "F1": f1_score(y_test, y_test_pred),
             "ROC AUC": roc_auc_score(y_test, y_test_pred_proba)
        }
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics
    
    def evaluate_validation(
            self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Parameters:
        -----------
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation features.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_val, y_val_pred),
            "Precision": precision_score(y_val, y_val_pred),
            "Recall": recall_score(y_val, y_val_pred),
            "F1": f1_score(y_val, y_val_pred),
            "ROC AUC": roc_auc_score(y_val, y_val_pred_proba)
        }
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics
    
    def run(self, df: pd.DataFrame):
        logging.info("Starting ML training pipeline...")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)

        # Baseline models
        baseline_models, baseline_metrics = self.train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val)

        # Tuned models
        tuned_models, tuned_metrics = self.train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val)

        # Combine results
        all_models = {**baseline_models, **tuned_models}
        all_metrics = {**baseline_metrics, **tuned_metrics}

        # Select best model
        best_model_name = max(
            all_metrics,
            key=lambda k: (
                0.7 * all_metrics[k]["F1"] +
                0.3 * all_metrics[k]["Accuracy"])
        )
 
        best_model = all_models[best_model_name]
        logging.info(f"Best model selected: {best_model_name}")

        # Final evaluation
        final_metrics = self.evaluate_final_model(
            best_model, X_test, y_test, best_model_name
        )

        logging.info(f"Final metrics: {final_metrics}")

        return {
            "model": best_model,
            "metrics": final_metrics,
            "name": best_model_name
        }