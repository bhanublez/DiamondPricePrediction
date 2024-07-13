import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info('Starting model training process...\n' + '-'*80)
            print("\nStarting model training process...\n" + "-"*80)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            print('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            print("\nModel Evaluation Report:")
            print("="*80)
            for model_name, r2_score in model_report.items():
                print(f"{model_name:20} : R2 Score = {r2_score:.4f}")

            logging.info(f'Model Report: {model_report}')

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f'\n{"Best Model Found":^80}')
            print("="*80)
            print(f'Best Model Name    : {best_model_name}')
            print(f'R2 Score           : {best_model_score:.4f}')
            print("="*80)

            logging.info(f'Best Model Found - Model Name: {best_model_name}, R2 Score: {best_model_score:.4f}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f'\nModel saved to: {self.model_trainer_config.trained_model_file_path}')
            logging.info(f'Model saved to: {self.model_trainer_config.trained_model_file_path}')

        except Exception as e:
            error_message = f"Exception occurred during Model Training: {str(e)}"
            logging.error(error_message)
            print(f'\n{error_message}\n' + '-'*80)
            raise CustomException(e, sys)
