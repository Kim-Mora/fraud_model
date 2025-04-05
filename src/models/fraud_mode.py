from pycaret.classification import *
from pandas import DataFrame
from typing import Any, Dict
from src.visualization.plots import create_report_for_best_model_on_pycaret_benchmarc 
from src.utils import get_configuration
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

CONFIG = get_configuration()

def create_tuned_model(train_set:DataFrame, model_type:str,
                             config: Dict = CONFIG)-> Any:
    
    clf_setup = setup(
    data=train_set.drop(config['model']['drop_features'], axis=1),
    target=config['model']['target'],              
    session_id=123,              
    normalize=True,                
    fix_imbalance=True,
    fix_imbalance_method=config['model']['balance_class_type'],
    )
    
    best_model = create_model(model_type)

    create_report_for_best_model_on_pycaret_benchmarc(best_model=best_model,
                                                      model_type=model_type)
    model = finalize_model(best_model)

    save_model(model, f'{model_type}')

def build_base_model(train_set:DataFrame, config:Dict= CONFIG)-> LogisticRegression:
    X = train_set.drop(config['target'])
    y = train_set[config['target']] 
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model
