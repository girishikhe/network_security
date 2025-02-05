from networksecurity.entity.config_entity import ModelEvaluationConfig
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants import TARGET_COLUMN
from networksecurity.logging.logger import logging
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from networksecurity.entity.estimator import NetworkModel, TargetValueMapping
from networksecurity.utils.main_utils import load_object, save_model # Assuming a function to load models from local storage
import os



@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float 

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

    def get_best_model(self) -> Optional[NetworkModel]:
        """
        Method Name :   get_best_model
        Description :   This function loads the best existing model from local storage.
        """
        try:
            model_path = self.model_eval_config.best_model_path
            if model_path is None or not os.path.exists(model_path):
                logging.warning("No previous best model found. Skipping evaluation.")
                return None
            return load_object(file_path=model_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   Evaluates the trained model against an existing model (if any).
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained_model = trained_model.predict(x)
            trained_model_f1_score = f1_score(y, y_hat_trained_model, average='weighted')

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model, average='weighted')
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Evaluation result: {result}")
            return result
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   Orchestrates the model evaluation process.
        """  
        try:
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_path=self.model_eval_config.best_model_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
