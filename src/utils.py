import pandas as pd
from pandas import Series
from geopy.distance import geodesic
from typing import Any, Str, Bool
from PIL import Image
import glob


def calculate_distance(row:Series, same_merchant:Bool=False)->Any:
    prevlat = 'prev_lat'
    prevlong = 'prev_long'

    if pd.isna(row[prevlat]) or pd.isna(row[prevlong]):
        return 0.0
    curr_point = (row['latitude'], row['longitude'])
    prev_point = (row[prevlat], row[prevlong])
    return geodesic(curr_point, prev_point).kilometers

def generate_report_with_existent_plots(path:Str)-> None:
    plots = sorted(glob.glob("*.png"))
    loaded_plots = [Image.open(img).convert("RGB") for img in plots]
    if loaded_plots:
        loaded_plots[0].saver_report(path, save_all=True,
                                     append_images=loaded_plots[1:]) 