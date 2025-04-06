import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame
from geopy.distance import geodesic

from PIL import Image
import glob

def get_configuration(path: str='config.yaml')-> dict:
    """
    Load the configuration file
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_report_with_existent_plots(path:str)-> None:
    plots = sorted(glob.glob("*.png"))
    loaded_plots = [Image.open(img).convert("RGB") for img in plots]
    if loaded_plots:
        loaded_plots[0].save(path, save_all=True,
                                     append_images=loaded_plots[1:]) 


def calculate_distance(row:Series)->float:
    prevlat = 'prev_lat'
    prevlong = 'prev_long'

    if pd.isna(row[prevlat]) or pd.isna(row[prevlong]):
        return 0.0
    curr_point = (row['latitude'], row['longitude'])
    prev_point = (row[prevlat], row[prevlong])
    return geodesic(curr_point, prev_point).kilometers


def calculate_woes(train_set:DataFrame, feature:str, 
                   target:str, delta:float=0.5)-> dict:

    grouped = train_set.groupby(feature)[target].agg(['sum', 'count'])
    grouped['non_event'] = grouped['count'] - grouped['sum']

    total_event = grouped['sum'].sum()
    total_non_event = grouped['non_event'].sum()

    grouped['woe'] = np.log(
        ((grouped['non_event'] + delta) / (total_non_event + delta * len(grouped))) /
        ((grouped['sum'] + delta) / (total_event + delta * len(grouped)))
    )
    woe_dict = grouped['woe'].to_dict() 
    with open('woe_dict.woe' , 'w') as f:
        json.dump(woe_dict, f)
    
    return woe_dict


def split_train_test_set(df: DataFrame,
                         test_size:float=0.2) -> tuple[DataFrame, DataFrame]:
    train_set, test_set = train_test_split(df, test_size=test_size,
                                        shuffle=False)
    return train_set, test_set


def build_cut_table(df: DataFrame, score_col: str = 'score',
                    target_col: str = 'Class', amount_col: str = 'Amount',
                    bins: int = 20) -> DataFrame:
    """
    Build a cut table with the follow information:
    - score threshold (cut)
    - acceptance_rate
    - number_of_fraud_accepted
    - number_of_fraud_rejected
    - acceptance_fraud_rate
    - rejected_fraud_rate
    - chargeback_rate
    - amount_lossed
    - amount_saved
    
    Args:
        df (DataFrame): dataframe with the whole population.
        score_col (str): Score Column.
        target_col (str): Target Column.
        amount_col (str): Amount Column
        bins (int): Number of bins to make the partition.

    Returns:
        DataFrame: Cut table. 
    """
    df = df.copy()


    bin_edges = np.linspace(0,1, bins+1)

    total_population = len(df)

    results = []

    for threshold in sorted(bin_edges[:-1], reverse=True):
        df_cut = df[df[score_col] <= threshold]
        df_ncut = df[df[score_col] > threshold]

        accepted_users = df_cut.shape[0]
        accepted_fraud = df_cut[df_cut.Class==1].shape[0]
        rejected_users = df_ncut.shape[0]
        rejected_fraud = df_ncut[df_ncut.Class==1].shape[0]
        acceptance_rate = (len(df_cut) / total_population) * 100
        acceptance_fraud_rate = (accepted_fraud /accepted_users) * 100 if accepted_users else 0
        rejected_fraud_rate = (rejected_fraud/ rejected_users) * 100 if rejected_users else 0
        
        chargeback_rate = df_cut[target_col].mean()
        amount_lossed = df_cut.loc[df_cut[target_col] == 1, amount_col].sum()
        amount_saved = df_ncut.loc[df_ncut[target_col] == 1, amount_col].sum()

        results.append({
            "cut": threshold,
            "acceptance_rate": acceptance_rate,
            "number_of_fraud_accepted": accepted_fraud,
            "number_of_fraud_rejected":rejected_fraud,
            "accepted_fraud_rate": acceptance_fraud_rate,
            "rejected_fraud_rate": rejected_fraud_rate,
            "chargeback_rate": chargeback_rate,
            "amount_lossed": amount_lossed,
            "amount_saved": amount_saved
        })

    return pd.DataFrame(results)
