import sys
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils import load_numpy_array_data, load_object, save_object
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from networksecurity.entity.estimator import NetworkModel



class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_and_evaluate_model(self, model, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Train a model and evaluate its performance on the test set.
        
        :param model: Scikit-learn model to be trained
        :param x_train: Training feature data
        :param y_train: Training labels
        :param x_test: Testing feature data
        :param y_test: Testing labels
        :return: Trained model and a ClassificationMetricArtifact containing evaluation metrics
        """
        try:
            model.fit(x_train, y_train)  # Train the model
            y_pred = model.predict(x_test)  # Predict on test set

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return model, metric_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_best_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Try multiple models and select the best one based on performance metrics.
        
        :param x_train: Training feature data
        :param y_train: Training labels
        :param x_test: Testing feature data
        :param y_test: Testing labels
        :return: Best model and its evaluation metrics
        """
        try:
            logging.info("Training models and selecting the best one.")
            
            # Initialize the models you want to try
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            best_model = None
            best_metric = None
            best_accuracy = 0

            # Iterate over models, train and evaluate them
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                trained_model, metric_artifact = self.train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
                accuracy = accuracy_score(y_test, trained_model.predict(x_test))
                
                # Select the best model based on accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = trained_model
                    best_metric = metric_artifact
            
            if best_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception("No model found with accuracy greater than expected base accuracy.")
            
            return best_model, best_metric
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate the model training process.
        
        :return: ModelTrainerArtifact containing paths to trained model and metrics
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Load transformed training and testing data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            # Split data into features and labels
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            # Get the best model
            best_model, metric_artifact = self.get_best_model(x_train, y_train, x_test, y_test)
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            # Save the trained model along with preprocessing object
            network_model = NetworkModel(preprocessing_object=preprocessing_obj, trained_model_object=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, network_model)
            
            # Create the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
