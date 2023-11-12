# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from dataclasses import dataclass

# Importing the models
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error as MAE, mean_squared_error as MSE
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

# Import custom modules for logging, exceptions, and utility functions
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define a dataclass for ModelTrainer configuration
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('achive', 'model.pkl')

# ModelTrainer class for training and evaluating machine learning models
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log that model training has started
            logging.info("Starting model training")

            # Split the training and test data arrays into input features and target variables
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define a dictionary of machine learning models to evaluate
            models = {
                'K_Neighbors clasifier': KNeighborsRegressor(),
                'DecisionTree Regressor': DecisionTreeRegressor(),
                'RandomForest Regressor': RandomForestRegressor(),
                'AdaBoost classifier': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'Ridge':Ridge() ,
                'Lasso':Lasso(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'Xgboost': XGBRegressor()
            }

            params= {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'alpha': [0.1, 1.0, 10.0]},
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                 
            }
            
            # Evaluate the models using the evaluate_model function
            model_report = evaluate_model(X_train, X_test, y_train, y_test, models, params)
            
            # Obtain the best model name based on the evaluation results
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get the best model from the dictionary
            best_model = models[best_model_name]

            # If the best model's score is below a threshold, raise a CustomException
            if best_model_score < 0.6:
                raise CustomException('No best model found')

            # Log that the best model was found on both training and test datasets
            logging.info("Best model found on both training datasets")

            # Save the best model to a specified file path
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions using the best model on the test data
            predicted = best_model.predict(X_test)

            # Calculate and return the R-squared (R2) score
            r2_square = r2_score(y_test, predicted)
            round(r2_square, 3)

            #calculate MSE
            mse = MSE(y_test, predicted)
            round(mse, 3)
            #calculate MAE
            
            mae = MAE(y_test, predicted)
            round(mae, 3)

            return {'best model name':best_model_name, "r2_square":r2_square , "MSE":MSE, "MAE":MAE}

        except Exception as e:
            # Raise a CustomException if an error occurs during model training
            raise CustomException(e, sys)
