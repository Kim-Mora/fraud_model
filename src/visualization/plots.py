import os
from pathlib import Path
from pandas import DataFrame
import sweetviz as sv

ROOT_DIR = Path.cwd().parents[0]

def get_eda_html(file_name:str, data:DataFrame) -> None:
    """Generate a Exploratory Data Analysis using 
    the sweetviz library. It give us visualizations of:
    - Columns Information.
    - linear correlation between columns
    - linear correlation with the response variable
    among others.

    Args:
        file_name (str): name to save the report.
        data_path (str): path to the data to be analized.
    """
    report = sv.analyze(source=data)
    report_path = os.path.join(ROOT_DIR, 'reports', file_name)
    report.show_html(filepath=report_path, open_browser=False)
