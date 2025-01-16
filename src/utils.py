import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd

import numpy as np
import dill
import pickle
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir = os.path.dirname(file_path)

        os.makedirs(dir, exist_ok=True)

        with open(file_path, 'wb') as f  :
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)

    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i, (model_name, model) in enumerate(models.items()):
            para = param.get(model_name, {})
            print(f"Evaluating {model_name} with parameters: {para}")

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            print(f"Best parameters for {model_name}: {gs.best_params_}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"{model_name} - Train Score: {train_model_score}, Test Score: {test_model_score}")
            report[model_name] = test_model_score

        return report
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
