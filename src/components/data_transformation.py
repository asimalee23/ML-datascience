import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_cols = ['writing score', 'reading score']
            cat_cols = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f"Numerical columns: {num_cols}")
            logging.info(f"Categorical columns: {cat_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_col_name = 'math score'

            input_feature_train_df = train_df.drop(columns=[target_col_name])
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name])
            target_feature_test_df = test_df[target_col_name]

            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object")

            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessing object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error during data transformation")
            raise CustomException(e, sys)
