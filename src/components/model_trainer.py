import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging


from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
            try:
                # All code inside the function MUST be indented
                logging.info('splitting training and testing data')
                X_train, y_train, X_test, y_test = (
                    train_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, :-1],
                    test_array[:, -1]
                )

                models = {
                    'LR': LinearRegression(),
                    'XGB': XGBRegressor(),
                    'GradientBoostingRegressor': GradientBoostingRegressor(),
                    'KNeighborsRegressor': KNeighborsRegressor(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(),
                    'RandomForestRegressor': RandomForestRegressor(),
                    'AdaBoostRegressor': AdaBoostRegressor()
                }

                params = {
                     "LR": {
                         "fit_intercept": [True, False],
                         "positive": [True, False]
                },
                  "XGB": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.3],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 1.5, 2]
                },

                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.6, 0.8, 1.0]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"],
                    "p": [1, 2]
                },

                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                    "max_features": ["sqrt", "log2", None]
                },

                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False]
                },

                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1],
                    "loss": ["linear", "square", "exponential"]
                }
                
                }


                
                
                model_report: dict = evaluate_models(
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    models=models, param=params
                )
                
                # Simplified getting the max score
                best_model_score = max(model_report.values())

                # Get the name of the best model
                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]

                best_model = models[best_model_name]

            
                
                logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted = best_model.predict(X_test)
                r2 = r2_score(y_test, predicted)
                return r2
            
            except Exception as e:
                raise CustomException(e, sys)