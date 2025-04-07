import os
from src.utils import generate_report_with_existent_plots
from pathlib import Path
import pandas as pd
from pandas import DataFrame, Series
import sweetviz as sv
from pycaret.classification import *
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import shap

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 39 
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22   

plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 18 
plt.style.use('ggplot')

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
        data (DataFrame): data to be analized.
    """
    report = sv.analyze(source=data)
    report_path = os.path.join(ROOT_DIR, 'reports', file_name)
    report.show_html(filepath=report_path, open_browser=False)

def create_bivariate_plots(df:DataFrame, column:str,
                           bins:int=5) -> None:
    """Creates a bivariate plot which shows de distribution
    of the good and the bad users for a given bins of a given column.

    Args:
        df (DataFrame): Dataframe with the data.
        column (str): column to be compared. 
        bins (int, optional): number of cuts on a given feature. Defaults to 5.
    """
    df = df.copy()
    results = []
    
    all_fraud_users = df[df.Class==1].shape[0]
    all_nfraud_users = df[df.Class==0].shape[0]
    df['bins'], bin_edges = pd.qcut(df[column], q=bins, 
                                    retbins=True, duplicates='drop')


    grouped = df.groupby(['bins', 'Class']).size().unstack(fill_value=0)

    # Renombrar columnas
    grouped.columns = ['no_fraud_users', 'fraud_users']

    grouped_percent = pd.DataFrame({
        'fraud_users': (grouped['fraud_users'] / all_fraud_users) * 100,
        'no_fraud_users': (grouped['no_fraud_users'] / all_nfraud_users) * 100
    })

    fig, ax = plt.subplots(figsize=(20,10))
    ax = grouped_percent[['fraud_users', 'no_fraud_users']].plot(
        kind='bar',
        color=['crimson', 'steelblue'],
        ax=ax
    )

    ax.set_title(f'Bivariate analysis by {column}', fontsize=16)
    ax.set_xlabel(f'{column} bins', fontsize=12)
    ax.set_ylabel('% of total users (per class)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.show()


def create_violin_plots(data:DataFrame,
                        cat_column:str='merchant',
                        target_column:str='Class',
                        comparison_col:str= None) -> None:
    """Creates a violint plot dividing the class by merchant
    and shows the distribution of the class with a given feature.
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.violinplot(data=data, x=cat_column, y=comparison_col,
                   hue=target_column, split=True, ax=ax)
    ax.set_xlabel('Merchants')
    ax.set_ylabel(f'{comparison_col}')
    ax.set_title(f"""Distribution of frauds on {comparison_col}
                        by merchant""")
    ax.grid(True, linestyle='--', c='gray')
    fig.show()


def create_report_for_best_model_on_pycaret_benchmarc(best_model:any, 
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
        model_type (str): type of model
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

def create_comparison_metrics_report(models:dict, x_test:DataFrame,
                                       y_test:Series) ->None:

    """Generates a report with the comparison of the follow metrics:
    - AUC
    - AUCPR
    - PRECISION
    -RECALL
    for all the given models. Then it saves it on a pdf file. 

    Args: 
        models (dict): dict with the follow way
        x_test (DataFrame): test data
        y_test (Series): test data
    """
    
    metrics = ['auc', 'aucpr', 'precision', 'recall']
    metric_funcs = {
        'auc': roc_auc_score,
        'aucpr': average_precision_score,
        'precision': precision_score,
        'recall': recall_score
    }
    global REPORTS_DIR

    report_path = os.path.join(REPORTS_DIR, 'models_metrics_compariosn.pdf')
    
    with PdfPages(report_path) as pdf:
        for metric in metrics:
            scores = {}
            for name, model in models.items():
                y_scores = model.predict_proba(x_test)[:, 1]
                y_pred = model.predict(x_test)

                if metric in ['auc', 'aucpr']:
                    score = metric_funcs[metric](y_test, y_scores)
                else:
                    score = metric_funcs[metric](y_test, y_pred)

                scores[name] = score

            sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            sorted_names = list(sorted_scores.keys())
            sorted_values = list(sorted_scores.values())

            plt.figure(figsize=(12, 6))
            bars = plt.bar(sorted_names, sorted_values, color='steelblue', edgecolor='black')
            plt.ylabel(metric.upper(), fontsize=14)
            plt.title(f'Comparación de modelos por {metric.upper()}', fontsize=12)
            plt.ylim(0, 1)
            plt.xticks(rotation=45, fontsize=12)

            for bar, value in zip(bars, sorted_values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{value:.3f}", ha='center', va='bottom', fontsize=10)

            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            pdf.savefig()
        plt.close()



def create_acceptance_vs_chargeback_rate_plot(cut_table:DataFrame,
                                              fraud_limit:float=0.012)-> None:
    """Generates an acceptance rate vs chargeback rate plot
       with a given cut table. It also sets point where the
       accepted_fraud_rate is lower that a given fraud limit.

    Args:
        cut_table (DataFrame): Cut table with the follow columns:
        - acceptance_rate
        - chargeback_rate
        - amount_saved
        - amount_lossed
        - accepted_fraud_rate
        fraud_limit (float): threshold to be used.
    """
    
    x = cut_table['acceptance_rate']
    y = cut_table['chargeback_rate']

    labels = cut_table['amount_saved']

    best_threshold_idx = cut_table[cut_table.accepted_fraud_rate >= fraud_limit
                                   ]['accepted_fraud_rate'].idxmax()
    best_threshold = cut_table.loc[best_threshold_idx]

    plt.figure(figsize=(20,10))
    plt.scatter(x, y, edgecolors='black', s=75)

    for i in range(len(cut_table)):
        plt.annotate(
            f"${labels.iloc[i]:,.0f}",
            (x.iloc[i], y.iloc[i]),
            textcoords = "offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=13,
            color='darkgreen'
        )

    plt.scatter(
        best_threshold['acceptance_rate'],
        best_threshold['chargeback_rate'],
        c='red', s=100, edgecolors='k',
        label='Best threshold'
    )
    plt.annotate(
        f"* Best trade-off\n+${best_threshold['amount_saved']:,.0f}",
        (best_threshold['acceptance_rate'], best_threshold['chargeback_rate']),
        textcoords='offset points',
        xytext=(30, -10),
        ha='left',
        fontsize=11,
        c='red'
    )

    plt.xlabel('Acceptance Rate', fontsize=15)
    plt.ylabel('chargeback Rate', fontsize=15)
    plt.title("Acceptance Rate vs Chargeback Rate. Labels: amount saved by rejected fraud users", fontsize=20)
    plt.grid(True, linestyle='--', c='gray', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def create_shap_summary_plot(x:DataFrame, model:any)-> None:
    """Generates a shap summary plot in order to have explainability
       for a given model.

    Args:
        x (DataFrame): DataFrame with features.
        model (any): Pycaret pipeline with a trained model

    """

    preprocessor = model[:-1]
    x_transformed = preprocessor.transform(x)
    estimator = model.named_steps['actual_estimator']

    explainer = shap.Explainer(estimator, x_transformed)
    shap_values = explainer.shap_values(x_transformed, check_additivity=False)

    shap.summary_plot(shap_values, x_transformed, plot_type='dot',
                      max_display=30)
