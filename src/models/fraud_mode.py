import yaml
from pycaret.classification import *
from pandas import DataFrame
from numpy import array
from src.visualization.plots import create_report_for_best_model_on_pycaret_benchmarc 
from src.utils import get_configuration

CONFIG = get_configuration()

def create_tuned_model(train_set:DataFrame, model_type:str)-> None:
    """Create a pycaret experiment with a given model.
    Possible models:
    - xgboost
    - lr (logistic_regression)
    - catboost
    - lightgbm

    It also creates a metrics report for the builded model. 

    Args:
        train_set (DataFrame): model training data. 
        model_type (str): type of model to be build.
    """
    
    clf_setup = setup(
    data=train_set.drop(CONFIG['model']['drop_features'], axis=1),
    target=CONFIG['model']['target'],              
    session_id=123,              
    normalize=True,                
    fix_imbalance=True,
    fix_imbalance_method=CONFIG['model']['balance_class_type'],
    )
    
    best_model = create_model(model_type)

    create_report_for_best_model_on_pycaret_benchmarc(best_model=best_model,
                                                      model_type=model_type)
    model = finalize_model(best_model)

    save_model(model, f'{model_type}')

def score_population(population:DataFrame, model:any) -> array:
    """Scores a given population with a given model.
    The model needs to be builded with pycaret in order to 
    get his features.

    Args:
        population (DataFrame): population to be scored.
        model (any): model to score the population.

    Returns:
        array: scores calculed using predict_proba function. 
    """
    model_features = model.feature_names_in_

    target_idx = model_features.index(CONFIG['model']['target'])

    model_features.pop(target_idx)

    scores = model.predict_proba(population[model_features])[:,1]
    return scores

def save_final_features(model:any) -> None:
    """Save the features for the best model on a yaml file.

    Args:
        model (any): model to save features.
    """
    model_features = model.feature_names_in_
    target_idx = model_features.index(CONFIG['model']['target'])
    model_features.pop(target_idx)
    
    with open('final_features.yml', 'w') as f:
        yaml.safe_dump(model_features, f)
