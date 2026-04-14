# Lung Cancer Risk Prediction with machine learning model.
### This project focuses on analysing the possibility of lung cancer occurrence based on data obtained from a reliable source. Features which holds a higher correlation to the occurrence of lung cancer will be used to build the predictive model. The model will be used to assess if a person is at high risk for lung cancer.
#### Side Note: This is a mini project used to serve as a prototype for various machine learning models and should not be used to fully define the risks of lung cancer.

* Prerequisites:
##### Setup environment using python version 3.11.10 and download dependencies stated in requirements.txt

* Pipeline execution
##### 1) In main.py file edit configure path 'config_path' to run config.yaml file in your designated project folder.
##### 2) In config.yaml file edit 'file_path' to read csv file in your designated project folder.
##### 3) Editing of parameters will be made in 'param_grid' if required.
##### 4) If new features are introduced into the dataset, edit 'file_path' accordingly and insert the new features into the feature lists accordingly.

* Logical Flow
##### 1) Load the .yaml file with all the dependencies as well as file path.
##### 2) Read the csv file into a dataframe.
##### 3) Initialize and run the data preparation class built in data_prep.py which consists of preprocessing steps, data has been cleaned beforehand hence it is not required.
##### 4) Initialize the model training class built in model_training.py.
##### 5) Split the data according to training(80%), testing(10%) and validation(10%) sets.
##### 6) Train and evaluate baseline models using XGBoost, Catboost, Random Forest and Decision Tree Classifier algorithms.
##### 7) Train and evaluate top performing baseline model after performing hyperparamter tuning.
##### 8) Combine all models and their metrics into dictionaries.
##### 9) Find the best model based on F1 and Accuracy scores.
##### 10) Evaluate the performance of the best model on the test set.

* Key Findings from EDA
##### 1) Majority of the data features are categorical in nature.
##### 2) There is a weak correlation between the features and target feature in this dataset.
##### 3) A change in weight holds the highest correlation with target feature.
##### 4) Patients with lung cancer history experiences higher increase in weight than weight loss.
##### 5) Lung cancer occurs more in Females than Males even though there're slightly more Males than Females in this dataset.
##### 6) Lung cancer occurs more in people with genetic histories in the family line.
##### 7) Dataset consists a higher ratio of elderly patients starting from age 60.
##### 8) Features gender, gene_markers, weight_change, smoking_duration, tiredness_frequency, air_pollution_level and age are highly significant to the target variable upon conducting three-way anova tests.

* Feature handling description
#
|         Feature        |                           Data handling Description                             | 
|------------------------|---------------------------------------------------------------------------------|
| All                    | Column labels are renamed to lower-caps for easier access and readability.      |
| Categorical            | Values are standardized with mapping to replace non-standardized data.          |
| COPD History           | Missing values replaced with 'No'.                                              |
| Taken Bronchodilators  | Missing values replaced with 'No'.                                              |
| Air Pollution Exposure | Missing values replaced with 'Low'.                                             |
| Lung Cancer Occurrence | Data type changed to 'object'.                                                  |
| Start Smoking          | Data type changed to 'int', dropped after feature engineering.                  |
| Stop Smoking           | Data type changed to 'int', dropped after feature engineering.                  |
| Gender                 | Value 'NAN' replaced with nonetype data and dropped.                            |
| ID                     | Deemed as irrelevant and dropped.                                               |
| Age                    | Absolute function applied to remove all negative values.                        |
| weight_change          | Feature engineered with features 'Last Weight' and 'Current Weight'.            |
| smoking_duration       | Feature engineered with features 'Stop Smoking' and 'Start Smoking'.            |
| Last Weight            | Dropped after feature engineering.                                              |
| Current Weight         | Dropped after feature engineering                                               |


* Model selection
#
|                  Model                  |                    Baseline Model description                       |                   Reason for selecting model                    |
|-----------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------|
| Decision Tree Classifier                | Tree-like structure, root node consists of entire dataset.          | Easy to interpret, handles numerical and categorical data.      |
| Categorical Boosting Classifier         | Designed specifically to handle categorical features, ensemble of trees using boosting method  | Datasets consists of categorical features majorly, robust to noisy data.|
| Extreme Gradient Boosting Classifier    | Optimized for speed and performance, ensemble of trees as above.    | Performs well with large datasets and highly customizable.      |
| Random Forest Classifier                | Ensemble of trees using bagging method, bootstrap sampling.         | Prevents overfitting, handles large datasets, reduces variance. |

* Baseline model evaluation on validation set
#
|       Model (Classifier)          |    Accuracy    |    Precision    |     Recall     |        F1        |       ROC AUC      |
|-----------------------------------|----------------|-----------------|----------------|------------------|--------------------|
| Decision Tree                     | 0.666307       | 0.695564        | 0.684523       | 0.69             | 0.666162           |
| Catboost                          | 0.735199       | 0.726315        | 0.821428       | 0.770949         | 0.842936           |
| XGBoost                           | 0.727664       | 0.733705        | 0.781746       | 0.756964         | 0.833888           |
| Random Forest                     | 0.730893       | 0.739622        | 0.777777       | 0.758220         | 0.823842           |
##### Top performing baseline model: Catboost(Categorical Boosting Classifier) is chosen based on F1 and Accuracy scores.
* Tuned model evaluation on validation set
#
|       Model (Classifier)          |    Accuracy    |    Precision    |     Recall     |        F1        |       ROC AUC      |
|-----------------------------------|----------------|-----------------|----------------|------------------|--------------------|
| Catboost                          | 0.710441       | 0.663877        | 0.944444       | 0.779688         | 0.821304           |
* Final evaluation on test set for catboost model
#
|       Model (Classifier)          |    Accuracy    |    Precision    |     Recall     |        F1        |       ROC AUC      |
|-----------------------------------|----------------|-----------------|----------------|------------------|--------------------|
| Catboost                          | 0.736275       | 0.711956        | 0.820459       | 0.762366         | 0.841067           |
##### Final results for Catboost model's performance on test set is within desired expectations, hence model will be deployed.

* Additional considerations to take before deployment:
#
##### 1) Changes in data distribution over time which might degrade model performance.
##### 2) Sufficiency of computational resources.
##### 3) Predictive speed of model in real-time applications.
##### 4) Deployment environment.
##### 5) User experience and business alignment.
