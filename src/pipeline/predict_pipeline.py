import os
import sys
import pandas as pd
from dataclasses import dataclass
from flask import request

from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN
from src.utils.main_utils import MainUtils


@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predicted_file.csv"

    @property
    def prediction_file_path(self):
        return os.path.join(self.prediction_output_dirname, self.prediction_file_name)


class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()

    def save_input_files(self) -> str:
        """
        Saves the uploaded CSV file to the prediction_artifacts directory.
        """
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)
            logging.info(f"Saved input CSV at {pred_file_path}")

            return pred_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Downloads model from S3, loads it, and predicts.
        """
        try:
            # make a copy and remove any columns that were not present at training time
            features_for_pred = features.copy()

            # Drop target column if present (prevents unseen feature errors)
            if TARGET_COLUMN in features_for_pred.columns:
                features_for_pred = features_for_pred.drop(columns=[TARGET_COLUMN])
                logging.info(f"Dropped target column '{TARGET_COLUMN}' before prediction.")

            # Drop common unwanted columns that sometimes appear when reading CSVs
            unnamed_cols = [c for c in features_for_pred.columns if str(c).startswith('Unnamed')]
            if unnamed_cols:
                features_for_pred = features_for_pred.drop(columns=unnamed_cols)
                logging.info(f"Dropped unnamed columns before prediction: {unnamed_cols}")

            model_local_dir = "artifacts"
            os.makedirs(model_local_dir, exist_ok=True)

            model_local_path = os.path.join(model_local_dir, "model.pkl")

            # Download model from S3
            model_path = self.utils.download_model(
                bucket_name="phishingclassifer",
                bucket_file_name="model.pkl",
                dest_file_name=model_local_path,
            )
            logging.info(f"Model downloaded successfully from S3: {model_path}")

            # Load model
            model = self.utils.load_object(file_path=model_path)
            # If possible, align features to the preprocessing object's expected input names
            try:
                preprocessor = getattr(model, 'preprocessing_object', None)
                expected_features = None

                if preprocessor is not None:
                    # sklearn >=1.0: many transformers expose feature_names_in_ or get_feature_names_out
                    if hasattr(preprocessor, 'feature_names_in_'):
                        expected_features = list(preprocessor.feature_names_in_)
                    else:
                        # try to call get_feature_names_out on the preprocessor or on the columns transformer
                        try:
                            expected_features = list(preprocessor.get_feature_names_out())
                        except Exception:
                            expected_features = None

                if expected_features is not None:
                    # select and reorder columns to match expected features, fill missing with NaN
                    missing = [c for c in expected_features if c not in features_for_pred.columns]
                    if missing:
                        logging.warning(f"Missing columns for model input: {missing}. They will be filled with NaN.")
                    aligned = features_for_pred.reindex(columns=expected_features)
                    features_for_pred = aligned

            except Exception:
                # if anything goes wrong while aligning, continue with the cleaned frame
                logging.info("Could not align input columns to preprocessor; proceeding with cleaned DataFrame.")

            # Predict using cleaned (and possibly aligned) features
            preds = model.predict(features_for_pred)
            logging.info("Prediction completed.")

            return preds

        except Exception as e:
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        """
        Reads input CSV, makes predictions, and saves a new CSV with predictions.
        """
        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            logging.info("Input CSV read successfully for prediction.")

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = predictions

            # Map numeric predictions to labels
            target_column_mapping = {0: 'phishing', 1: 'safe'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index=False)

            logging.info(f"Predictions saved at {self.prediction_file_detail.prediction_file_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        """
        Executes the entire prediction pipeline.
        """
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            return self.prediction_file_detail
        except Exception as e:
            raise CustomException(e, sys)
