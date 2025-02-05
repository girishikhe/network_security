import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Optional
from dataclasses import dataclass

from networksecurity.entity.config_entity import ModelEvaluationConfig
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from networksecurity.entity.estimator import NetworkModel
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants import TARGET_COLUMN
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import load_object

import dagshub
dagshub.init(repo_owner='girishikhe', repo_name='network_security', mlflow=True)


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

def track_mlflow(best_model, metric_artifact):
    """
    Logs model performance metrics and registers the best model in MLflow.
    """
    try:
        with mlflow.start_run():
            mlflow.log_metric("f1_score", metric_artifact.f1_score)
            mlflow.log_metric("precision", metric_artifact.precision_score)
            mlflow.log_metric("recall", metric_artifact.recall_score)
            mlflow.sklearn.log_model(best_model, "model")
            
            tracking_url_type_store = mlflow.get_tracking_uri()
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best_Model")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

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
        """Loads the best existing model from local storage."""
        try:
            model_path = self.model_eval_config.best_model_path
            if model_path is None or not os.path.exists(model_path):
                logging.warning("No previous best model found. Skipping evaluation.")
                return None
            return load_object(file_path=model_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """Evaluates the trained model against an existing best model."""
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            #y = y.replace(TargetValueMapping()._asdict())
            y = y.map({1: 1, -1: 0})

            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            y_hat_trained_model = trained_model.predict(x)
            trained_model_f1_score = f1_score(y, y_hat_trained_model)
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model, )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=trained_model_f1_score,
                precision_score=precision_score(y, y_hat_trained_model ),
                recall_score=recall_score(y, y_hat_trained_model)
            )
            track_mlflow(trained_model, metric_artifact)

            tmp_best_model_score = best_model_f1_score if best_model_f1_score is not None else 0
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
        """Orchestrates the model evaluation process."""
        try:
            evaluate_model_response = self.evaluate_model()

            best_model_path = self.model_trainer_artifact.trained_model_file_path if evaluate_model_response.is_model_accepted else self.model_eval_config.best_model_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_path=best_model_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
