import os
from typing import Any
from src.utils import generate_report_with_existent_plots
from pathlib import Path
from pandas import DataFrame
import sweetviz as sv
from pycaret.classification import *


ROOT_DIR = Path.cwd().parents[0]
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')

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


def create_report_for_best_model_on_pycaret_benchmarc(best_model:Any, 
                                                      model_type:str)-> None:
    """Generate a report for the model with the best performance on the
    pycaret benchmark whit the follow metrics:
    - AUC
    - Confusion Matrix
    - AUCPR
    - Feature Importance
    - Class Report

    Args:
        best_model (Any): best_model on pycaret benchmark
    """
    
    plot_types = ['confusion_matrix','auc',
                'pr', 'feature', 
                'class_report']
    
    for plot in plot_types:
        try:
            plot_model(best_model, plot=plot, save=True)
        except:
            print(f"Error generating: {plot}")

    report_name = f'{model_type}_report.pdf'
    report_path = os.path.join(REPORTS_DIR, report_name)
    generate_report_with_existent_plots(report_path)
