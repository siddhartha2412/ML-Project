""" For common functions across the project , such as data base connections or  saving models, etc."""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object into a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Evaluate multiple models with GridSearchCV and return a score report.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models: Dictionary of model_name -> model_object
        param: Dictionary of model_name -> param_grid
    Returns:
        report (dict): model_name -> best test score
    """
    try:
        report = {}

        for model_name in models.keys():
            model = models[model_name]
            params = param[model_name]

            gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "best_params": gs.best_params_,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "mae": mean_absolute_error(y_test, y_test_pred),
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
