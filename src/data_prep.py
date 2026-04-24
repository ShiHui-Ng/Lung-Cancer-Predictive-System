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
        
<<<<<<< HEAD
        return ColumnTransformer([
            ("nom", nominal_transformer, nominal_features),
            ("ord", ordinal_transformer, ordinal_features)
        ], remainder="passthrough")
=======
        if not isinstance(features, list):
            raise ValueError("Features must be provided as a list of column names")
    
        value_counts_dict = {}

        for feature in features:
            if feature in data.columns:
                #Apply value_counts with dropna=False to retain None/NaN
                value_counts_dict[feature] = data[feature].value_counts(
                    normalize=False, dropna=False
                    )
            else:
                raise KeyError(f"Feature '{feature} not found in DataFrame")
        
        return value_counts_dict
    
    @staticmethod
    def bar_plot(cols, rows, df, plot):
        '''
        Plot bar charts to examine data distribution across binary and numerical features.

        Parameters:
        -----------
        - cols (Variable): Number of columns showcasing the feature plots in the pane
        - rows (Variable): Number of rows showcasing the feature plots in the pane
        - df (pd.DataFrame): The input DataFrame
        - plot (function): The type of plot to use for the selected features 

        Returns:
        --------
        - A single pane containing multiple bar plots showcasing various data distributions of binary and numerical features

        '''
        fig = plt.figure(figsize = (cols*5, rows*5))

        for i, col in enumerate(df):
            ax = fig.add_subplot(rows, cols, i + 1)
            plot(x = data[col], ax = ax)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def horizontal_bar_plot(cols, rows, df, plot):
        '''
        Plot bar charts to examine data distribution across binary and numerical features.

        Parameters:
        -----------
        - cols (Variable): Number of columns showcasing the feature plots in the pane
        - rows (Variable): Number of rows showcasing the feature plots in the pane
        - df (pd.DataFrame): The input DataFrame
        - plot (function): The type of plot to use for the selected features 

        Returns:
        --------
        - A single pane containing multiple bar plots showcasing various data distributions of binary and numerical features

        '''
        fig = plt.figure(figsize = (cols*5, rows*5))

        for i, col in enumerate(df):
            ax = fig.add_subplot(rows, cols, i + 1)
            plot(y = data[col], ax = ax)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def outlier_detection(data, features, plot):
        '''
        Detect outliers for multiple features in a single visualization pane
        
        Parameters:
        -----------
        - data (DataFrame): The DataFrame to use for plotting graphs.
        - features (variables): Input feature(s) in DataFrame
        - plot (function): The type of plot to use for input features

        Returns:
        --------
        Multiple plots in a single pane
        '''

        fig = plt.figure(figsize=(10,6))
        plot(data[features])
        fig.tight_layout()
        plt.title('Outliers Detection')
        plt.show()

    @staticmethod
    def corr_analysis(features, corr_method, title):
        '''
        Plot heatmap of correlation matrix for selected features in dataset

        Parameters:
        -----------
        - features (variables): Input feature(s) in DataFrame
        - corr_method (function): The type of correlation method used to analysis relationship between selected features
        - title (string): The title of the correlation analysis

        Returns:
        --------
        Heatmap matrix showcasing correlation figures between selected features 
        '''
        #Select continuous variables
        corr_vars_all = data1[features]

        #Calculate the correlation matrix for selected features
        corr_matrix = corr_vars_all.corr(method=corr_method)

        #Plot heatmap of correlation matrix for selected features
        plt.figure(figsize=(10,6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, center=0.5, vmax=1,
                    linewidths=0.5, fmt=".2f")
        plt.title(title)
        plt.show()

    @staticmethod
    def factorize_columns(cols):
        '''
        Encode string values with unique integers across multiple categorical features

        Parameters:
        -----------
        - cols (variables): Columns(features) extracted from the DataFrame

        Returns:
        --------
        Features with encoded data
        '''
        for col in cols:
            data1[col], _ = pd.factorize(data1[col])
        return data1
    
    @staticmethod
    def compute_pairwise_mcc(columns):
        '''
        Compute Matthews Correlation Coefficient (MCC) for all pairs of binary features.

        Parameters:
        -----------
        - columns (variables): Features in DataFrame

        Returns:
        --------
        dict: Pairwise MCC for each pair of binary features.
        '''
    
        mcc_results = {}

        # Loop through all unique pairs of columns
        for col1, col2 in itertools.combinations(columns, 2):
            # Ensure binary features are in 0/1 format
            if sorted(data1[col1].unique()) == [0, 1] and sorted(data1[col2].unique()) == [0, 1]:
                mcc = matthews_corrcoef(data1[col1], data1[col2])
                mcc_results[(col1, col2)] = mcc
            else:
                raise ValueError(f"One of the columns ({col1} or {col2}) is not binary (0/1).")

        return mcc_results
    
    @staticmethod
    def pbiserial_coeff(cont_features, binary_features, robust):
        '''
        Compute Point-Biserial coefficient between continuous and binary features.

        Parameters:
        -----------
        - continuous_features (variable): Continuous features in the DataFrame to be analysed.
        - binary_features (variable): Binary features in the DataFrame to be analysed.

        Returns:
        --------
        Correlation coefficient and p-value between each pair of continuous and binary features.
        '''
        X = data1[cont_features]
        y = data1[binary_features]

        # Initialise results dictionary
        corr_results = {}
        # Iterate over features and calculate point-biserial correlation
        for feature in X:
            corr, p_value = pointbiserialr(y, data1[feature])
            corr_results[feature] = {'Correlation': corr, 'P_value': p_value}

        # Display results
        for feature, stats in corr_results.items():
            print(f"Feature: {feature}")
            print(f"  Correlation: {stats['Correlation']:.2f}")
            print(f"  P-value: {stats['P_value']:.4f}")


    @staticmethod
    def scatter(feature, target, hue):
        '''
        Plot scatterplot with jitter for vizualising relationships between variables.

        Parameters:
        -----------
        - feature (str or Series): The main variable to plot on the x-axis. Typically numeric or categorical.
        - hue (str or Series): A categorical or numeric variable to color the points, adding a third dimension to the scatterplot.

        Returns:
        --------
        None: Displays a scatterplot with jitter for enhanced vizualisation.
        '''

        # Add jitter to features
        feature1_jitter = data1[feature] + np.random.normal(0, 0.05, len(data1))
        target_feature = data1[target] + np.random.normal(0, 0.5, len(data1))

        # Plot scatterplot
        sns.scatterplot(data=data1, x=feature1_jitter, y=target_feature, hue=hue)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='10')
        plt.show() 

    @staticmethod
    def three_way_anova(dependent_var, var1, var2, var3):
        '''
        Build a three-way anova model to find the significance between 3 features and
        target feature.

        Parameters:
        -----------
        dependent_var (variable): Target binary variable
        var1 (variable): Binary feature column
        var2 (variable): Binary feature column
        var3 (variable): Continuous feature column

        Returns:
        --------
        ANOVA test model results
        '''
        # Fit three-way anova model
        model = ols(
            f'{dependent_var} ~ C({var1}) + C({var2}) + {var3} + '
            f'C({var1}):C({var2}) + C({var1}):{var3} + '
            f'C({var2}):{var3} + C({var1}):C({var2}):{var3}',
            data=data1
            ).fit()
    
        # Perform ANOVA test
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    
>>>>>>> 4c0272762fd503b86a8a69d49046228cb2f5c054
