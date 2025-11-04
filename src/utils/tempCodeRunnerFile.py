aticmethod
def download_model(bucket_name, bucket_file_name, dest_file_name):
    try:
        os.makedirs(os.path.dirname(dest_file_name), exist_ok=True)
        s3_client = boto3.client(
            "s3",
            aws_access_key_id="YOUR_ACCESS_KEY",
            aws_secret_access_key="YOUR_SECRET_KEY",
            region_name="ap-south-1"
        )
        s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)
        logging.info(f"Downloaded {bucket_file_name} from s3://{bucket_name} to {dest_file_name}")
        return dest_file_name
    except Exception as e:
        raise CustomException(e, sys)

@staticmethod
def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
    """
    Method Name: remove_unwanted_spaces
    Description: This method removes the unwanted spaces from a pandas dataframe.
    Output: A pandas DataFrame after removing the spaces.
    On Failure: Raise Exception

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None
    """
    try:
        df_without_spaces = data.apply(
            lambda x: x.str.strip() if x.dtype == "object" else x)
        logging.info(
            'Unwanted spaces removal Successful. Exited the remove_unwanted_spaces method of the Preprocessor class')
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
