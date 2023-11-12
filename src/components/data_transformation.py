from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('achive', "preprocessor.pkl")

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Perform one-hot encoding for each column
        encoded_cols = [pd.get_dummies(X[col], prefix=col, dtype=int) for col in X.columns]
        encoded_df = pd.concat(encoded_cols, axis=1)
        return encoded_df
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        numerical_columns = ['Year', 'exchange_rate', 'Population', 'Per_capita_GNI',
                            'Agriculture_hunting_forestry_fishing', 'Construction',
                            'Exports_of_goods_and_services', 'Final_consumption_expenditure',
                            'General_government_final_consumption_expenditure',
                            'Gross_capital_formation', 'Gross_fixed_capital_formation',
                            'Household_consumption_expenditure', 'Imports_of_goods_and_services',
                            'Manufacturing_index', 'Mining_Manufacturing_Utilities',
                            'Other_Activities', 'Total_Value_Added ',
                            'Transport_storage_and_communication',
                            'Wholesale_retail_trade_restaurants_and_hotels' ]
        categorical_columns = ['Country']

        # Numerical preprocessing pipeline
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline(
            steps=[
                ("encoder", CategoricalEncoder())
            ]
        )

        logging.info(f"numerical columns: {numerical_columns}")

        # Combining the pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", numeric_pipeline, numerical_columns),
                ("cat_pipeline", categorical_pipeline, categorical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data complete")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = 'Gross_Domestic_Product'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on the training and test dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved Preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
