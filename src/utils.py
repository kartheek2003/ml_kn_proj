import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd

import numpy as np
import dill


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir = os.path.dirname(file_path)

        os.makedirs(dir, exist_ok=True)

        with open(file_path, 'wb') as f  :
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)

    
