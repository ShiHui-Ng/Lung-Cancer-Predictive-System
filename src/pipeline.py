
import logging
import joblib
from data_prep import DataPrep
from model_training import ModelTraining
from data.data_loader import load_data

class MLPipeline:
    def __init__(self, config):
        self.config = config
        self.data_prep = DataPrep(config)
    
    def _load_data(self):
        return load_data(
            self.config["file_path"],
            f"SELECT * FROM {self.config['table_name']}"
            )
    
    def _clean_data(self, df):
        return self.data_prep.clean_data(df)

    def run(self):
        df = self._load_data()
        cleaned_df = self._clean_data(df)
        model_training = ModelTraining(self.config, self.data_prep.preprocessor)

        X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)

        baseline_models, baseline_metrics = model_training.train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val)
        tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val)

        all_models = {**baseline_models, **tuned_models}
        all_metrics = {**baseline_metrics, **tuned_metrics}

        best_model_name = max(
            all_metrics,
            key=lambda k: (
                0.7 * all_metrics[k]["F1"] +
                0.3 * all_metrics[k]["Accuracy"])
        )

        best_model = all_models[best_model_name]
        logging.info(f"Best model selected: {best_model_name}")

        # Final evaluation
        final_metrics = model_training.evaluate_test_set(
            best_model, X_test, y_test, best_model_name
        )

        logging.info(f"Final metrics: {final_metrics}")

        # Save model (CLEAN VERSION)
        model_path = self.config.get("model_output_path", "lung_cancer_model.pkl")
        joblib.dump(best_model, model_path)
        logging.info(f"Model saved to: {model_path}")

        # Return pipeline output
        return {
            "model": best_model,
            "metrics": final_metrics,
            "name": best_model_name,
            "model_path": model_path
        }