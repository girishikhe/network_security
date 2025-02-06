import os
import pandas as pd 
import numpy as np 
import sys

from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException 


DATABASE_NAME = "NetworkSecurity"

COLLECTION_NAME = "NetworkData"

MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "networksecurity"
ARTIFACT_DIR: str = "artifact"


TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "Phishing_data.csv"
MODEL_FILE_NAME: str = "model.pkl"


TARGET_COLUMN = "Result"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

SAVED_MODEL_DIR = os.path.join("saved_model")


"""
Data Ingestion related constant start with DATA INGESTION VAR NAME 
"""

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2



"""
Data Validation related constant starts with DATA VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related constant starts with DATA TRANSFORMTION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


"""
MODEL TRAINER related constant start with MODEL_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.8
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.07
#MODEL_BUCKET_NAME = "deposit-model2024"
#MODEL_PUSHER_S3_KEY = "model-registry"
MODEL_EVALUATION_BEST_MODEL_DIR = "model_evaluation"
MODEL_FILE_NAME = "best_model.pkl"

"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "usvisa-model2024"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000