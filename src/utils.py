import yaml
import pandas as pd
import numpy as np
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


def calculate_distance(row:Series)->float:
    prevlat = 'prev_lat'
    prevlong = 'prev_long'

    if pd.isna(row[prevlat]) or pd.isna(row[prevlong]):
        return 0.0
    curr_point = (row['latitude'], row['longitude'])
    prev_point = (row[prevlat], row[prevlong])
    return geodesic(curr_point, prev_point).kilometers

def generate_report_with_existent_plots(path:str)-> None:
    plots = sorted(glob.glob("*.png"))
    loaded_plots = [Image.open(img).convert("RGB") for img in plots]
    if loaded_plots:
        loaded_plots[0].save(path, save_all=True,
                                     append_images=loaded_plots[1:]) 


def build_separed_set(fusers, nfusers, df, size)-> any:
    fraud_users = fusers[:int(len(fusers) * size)]
    no_fraud_users = nfusers[:int(len(nfusers) * size)]
    separed_users = list(set(list(fraud_users) + list(no_fraud_users)))
    separed_set = df[df.credit_card_number.isin(separed_users)]
    return separed_set

def split_set(df, big_set_size):
    fraud_users = df[df.Class==1].credit_card_number.unique().tolist()
    no_fraud_users = df[df.Class==0].credit_card_number.unique().tolist()
    big_set = build_separed_set(fraud_users, no_fraud_users, 
                                df, big_set_size)
    small_set = df[~df.credit_card_number.isin(big_set.credit_card_number.tolist())]
    return big_set, small_set

def split_train_test_val_set(df, train_size, val_size) -> tuple[DataFrame, DataFrame, DataFrame]:
    train_set, test_set = split_set(df=df, big_set_size=train_size)
    test_set, val_set = split_set(df=test_set, big_set_size=val_size)
    return train_set, test_set, val_set


def build_cut_table(df:DataFrame, score_col:str='score', target_col:str='Class',
                    amount_col:str='Amount', bins:int = 20) -> DataFrame:

    df = df.copy()
    df.sort_values(by=score_col, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    thresholds = np.quantile(df[score_col], q=np.linspace(0, 1, bins + 1))[1:]

    total_population = len(df)
    total_chargeback = df[target_col].sum()
    total_amount_chargeback = df.loc[df[target_col] == 1, amount_col].sum()

    results = []
    for cut in thresholds:
        df_cut = df[df[score_col] >= cut]

        acceptance_rate = len(df_cut) / total_population
        chargeback_rate = df_cut[target_col].sum() / \
                    total_chargeback if total_chargeback > 0 else 0

        amount_lossed = df_cut.loc[df_cut[target_col] == 1, amount_col].sum()
        amount_saved = total_amount_chargeback - amount_lossed

        results.append({
            "cut": cut,
            "acceptance_rate": acceptance_rate,
            "chargeback_rate": chargeback_rate,
            "amount_lossed": amount_lossed,
            "amount_saved": amount_saved
        })

    return DataFrame(results)