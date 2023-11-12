# Import necessary libraries and modules
import os
from dataclasses import dataclass
import sys
from src.logger import logging  # Import the logging module from a custom source
from  src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Define a data class for configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('achive', 'train.csv')  # Default path for training data
    test_data_path: str = os.path.join('achive', 'test.csv')    # Default path for test data
    raw_data_path: str = os.path.join('achive', 'data.csv')     # Default path for raw data

# Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize the data ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log entry message

        try:
            # Read the dataset from a CSV file
            df = pd.read_csv('src/components/notebook/data to use.csv')
            logging.info("Read the dataset as a dataframe")  # Log message

            # Create directories if they don't exist for data paths
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the dataset to the raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")  # Log message

            # Split the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")  # Log message

            # Return the paths of the train and test data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Raise a custom exception with error details
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()  # Call the data ingestion method

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    results = model_trainer.initiate_model_trainer(train_arr, test_arr)

    # Print the results
    for key, value in results.items():
        print(f"{key}: {value}")
    
