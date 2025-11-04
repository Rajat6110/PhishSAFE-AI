import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Insert project path so imports can work if script is run from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.main_utils import MainUtils
from src.constant import TARGET_COLUMN


def load_model(model_path: str):
    utils = MainUtils()
    return utils.load_object(model_path)


def align_features(model, df: pd.DataFrame) -> pd.DataFrame:
    """Align input dataframe columns to the model's preprocessing object if possible."""
    df_clean = df.copy()
    if TARGET_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[TARGET_COLUMN])

    preprocessor = getattr(model, 'preprocessing_object', None)
    expected = None
    if preprocessor is not None:
        if hasattr(preprocessor, 'feature_names_in_'):
            expected = list(preprocessor.feature_names_in_)
        else:
            try:
                expected = list(preprocessor.get_feature_names_out())
            except Exception:
                expected = None

    if expected is not None:
        missing = [c for c in expected if c not in df_clean.columns]
        if missing:
            print(f"Warning: missing columns in test set: {missing}. Filling with NaN.")
        aligned = df_clean.reindex(columns=expected)
        return aligned

    # fallback: drop unnamed columns
    unnamed = [c for c in df_clean.columns if str(c).startswith('Unnamed')]
    if unnamed:
        df_clean = df_clean.drop(columns=unnamed)
    return df_clean


def evaluate(model_path: str, test_csv: str):
    model = load_model(model_path)
    df = pd.read_csv(test_csv)
    # If target column present -> evaluate. Otherwise produce predictions CSV.
    if TARGET_COLUMN in df.columns:
        y_true = df[TARGET_COLUMN].copy()
        X = df.drop(columns=[TARGET_COLUMN])

        X_aligned = align_features(model, X)

        y_pred = model.predict(X_aligned)

        # If predictions are numeric and target in test csv is strings like 'phishing'/'safe', map accordingly
        if y_true.dtype == object:
            try:
                mapping = {0: 'phishing', 1: 'safe'}
                y_pred_mapped = pd.Series(y_pred).map(mapping)
                if y_pred_mapped.isnull().all():
                    y_pred_final = y_pred
                else:
                    y_pred_final = y_pred_mapped
            except Exception:
                y_pred_final = y_pred
        else:
            y_pred_final = y_pred

        acc = accuracy_score(y_true, y_pred_final)
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_final))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred_final))
    else:
        # No labels: create predictions CSV and write to disk
        X_aligned = align_features(model, df)
        y_pred = model.predict(X_aligned)

        # map numeric preds to labels if reasonable
        try:
            mapping = {0: 'phishing', 1: 'safe'}
            y_pred_mapped = pd.Series(y_pred).map(mapping)
            if y_pred_mapped.isnull().all():
                out_preds = pd.Series(y_pred, name=TARGET_COLUMN)
            else:
                out_preds = y_pred_mapped.rename(TARGET_COLUMN)
        except Exception:
            out_preds = pd.Series(y_pred, name=TARGET_COLUMN)

        out_df = df.copy()
        out_df[TARGET_COLUMN] = out_preds.values

        out_path = os.path.join('predictions', os.path.basename(test_csv))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Predictions written to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join('artifacts', 'model.pkl'), help='Path to trained model file')
    parser.add_argument('--test', default=os.path.join('prediction_artifacts', 'phisingtest.csv'), help='Path to test CSV file')
    args = parser.parse_args()

    evaluate(args.model, args.test)
