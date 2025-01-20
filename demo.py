from networksecurity.pipline.training_pipeline import TrainPipeline
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig


import sys
from networksecurity.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()

