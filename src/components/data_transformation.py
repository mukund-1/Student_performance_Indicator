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
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_column=["writing_score", "reading_score"]
            categorical_column=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )
                
            cat_pipleline=Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]    
            )
            logging.info(f"Categorical columns: {categorical_column}")
            logging.info(f"Numerical columns: {numerical_column}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num pipeline", num_pipeline, numerical_column),
                    ("cat_pipeline", cat_pipleline, categorical_column)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_columns_nam = "math_score"
            numerical_column=["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_columns_nam], axis=1)
            target_feture_train_df = train_df[target_columns_nam]
            
            input_feature_test_df = test_df.drop(columns=[target_columns_nam], axis=1)
            target_feture_test_df = test_df[target_columns_nam]
            
            logging.info(f"Applying preprocessor object on training dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feture_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feture_test_df)]
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
        
    
