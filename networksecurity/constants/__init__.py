import os
import pandas as pd 
import numpy as numpy
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


TARGET_COLUMN = "Result"


"""
Data Ingestion related constant start with DATA INGESTION VAR NAME 
"""

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

