from pathlib import Path
import pandas as pd
from pandas import DataFrame

def load_data(path:str)->DataFrame:
    """Load a dataset with a given path.
    The file could be:
    - csv
    - parquet

    Args:
        path (str): path for a given dataset.
    --------
    Return:
        data (DataFrame): pandas DataFrame with 
        the loaded Data
    """

    extension = Path(path).suffix.lower()
    match extension:
        case '.csv':
            data = pd.read_csv(path)
            return data
        case '.parquet':
            data = pd.read_parquet(path=path)
            return data
        case _:
            raise ValueError("Unsuported Extension")

def save_data(path:str, data:DataFrame) -> None:
    """Save a given dataset on a given path.
    It supports:
    - csv
    - parquet

    Args:
        path (str): path of the new file.
        data (DataFrame): data to be saved.

    Raises:
        ValueError: uknown or unsupported file extension.
    """
    extension = Path(path).suffix.lower()
    match extension:
        case '.csv':
            data.to_csv(path)
        case '.parquet':
            data.to_parquet(path=path)
        case _:
            raise ValueError("Unsuported Extension")
        
    