# Standard;  library imports
import logging
import re
from typing import Any, Dict
import json

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt


class DataPrep:
    '''
    A class used to clean and preprocess lung cancer data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    '''

    def __init__(self, config):
        '''
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        '''
        self.config = config
        self.preprocessor = self.build_preprocessor()

    def clean_data(self, data, drop_duplicates=False):
        logging.info('Running data validation checks...')

        # Ensure correct schema
        expected_cols = self.config["numerical_features"] + \
                        self.config["nominal_features"] + \
                        self.config["ordinal_features"] + \
                        [self.config["target_column"]]
        
        missing_cols = set(expected_cols) - set(data.columns)
        extra_cols = set(data.columns) - set(expected_cols)

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        if extra_cols:
            logging.warning(f"Unexpected columns found: {extra_cols}")
        
        # Initial row tracking
        before = len(data)

        # Validate duplicates
        if drop_duplicates:
            before = len(data)
            data = data.drop_duplicates()
            logging.info(f"Dropped {before - len(data)} duplicates")
        
        # Missing values
        missing_ratio = data.isnull().mean().mean()
        logging.info(f"Missing value ratio: {missing_ratio:.2%}")
        
        data = data.dropna()

        # Final row tracking
        after = len(data)
        logging.info(f"Final rows:  {after} (removed {before - after})")

        logging.info("Data validation completed.")
        return data
    
    def build_preprocessor(self) -> ColumnTransformer:
        nominal_features = self.config["nominal_features"]
        ordinal_features = self.config["ordinal_features"]
        ordinal_categories = self.config["ordinal_categories"]

        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
            )
        ordinal_transformer = Pipeline(
            steps=[
                ("ordinal",
                 OrdinalEncoder(
                     categories=self.config["ordinal_categories"],
                     handle_unknown="use_encoded_value",
                     unknown_value=-1
                     ))])
        
        return ColumnTransformer([
            ("nom", nominal_transformer, nominal_features),
            ("ord", ordinal_transformer, ordinal_features)
        ], remainder="passthrough")