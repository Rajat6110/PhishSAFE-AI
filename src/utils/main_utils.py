# ...existing code...
import sys
from typing import Dict, Tuple
import os

import numpy as np
import pandas as pd
import pickle
import yaml
import boto3

from src.constant import *
from src.exception import CustomException
from src.logger import logging


class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))
            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            logging.info("Exited the save_object method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")
        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
            logging.info("Exited the load_object method of MainUtils class")
            return obj
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def upload_file(from_filename: str, to_filename: str, bucket_name: str):
        try:
            s3_resource = boto3.resource("s3")
            # boto3.upload_file signature: Filename, Bucket, Key
            s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def download_model(bucket_name: str, bucket_file_name: str, dest_file_name: str):
        """
        Download model to dest_file_name.
        Behavior for local usage (no AWS):
        - If dest_file_name already exists, return it immediately.
        - If bucket_name is falsy and local file missing -> raise FileNotFoundError.
        - Otherwise try to download from S3 (uses normal boto3 credential resolution).
        """
        try:
            # use local copy if available (works offline)
            if os.path.exists(dest_file_name):
                logging.info(f"Using local model at: {dest_file_name}")
                return dest_file_name

            # ensure target dir exists
            target_dir = os.path.dirname(dest_file_name) or "."
            os.makedirs(target_dir, exist_ok=True)

            # if no bucket specified, fail with clear message
            if not bucket_name:
                raise FileNotFoundError(f"Local model not found: {dest_file_name}")

            # attempt S3 download (will use environment/credentials chain if available)
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)
            logging.info(f"Downloaded {bucket_file_name} from s3://{bucket_name} to {dest_file_name}")
            return dest_file_name

        except Exception as e:
            # handle missing credentials gracefully if local file exists
            try:
                from botocore.exceptions import NoCredentialsError
            except Exception:
                NoCredentialsError = None

            if NoCredentialsError is not None and isinstance(e, NoCredentialsError):
                if os.path.exists(dest_file_name):
                    logging.info("AWS credentials missing — returning local model.")
                    return dest_file_name
                raise CustomException("AWS credentials not found and local model is missing.", sys) from e

            raise CustomException(e, sys) from e

    @staticmethod
    def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unwanted spaces from pandas dataframe.
        """
        try:
            df_without_spaces = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            logging.info('Unwanted spaces removal Successful.')
            return df_without_spaces
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def identify_feature_types(dataframe: pd.DataFrame):
        data_types = dataframe.dtypes

        categorical_features = []
        continuous_features = []
        discrete_features = []

        for column, dtype in dict(data_types).items():
            unique_values = dataframe[column].nunique()

            if dtype == 'object' or unique_values < 10:
                categorical_features.append(column)
            elif dtype in [np.int64, np.float64]:
                if unique_values > 20:
                    continuous_features.append(column)
                else:
                    discrete_features.append(column)
            else:
                pass

        return categorical_features, continuous_features, discrete_features
# ...existing code...