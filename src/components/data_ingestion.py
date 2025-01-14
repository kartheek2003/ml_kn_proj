import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.components.data_transformation import DataTransformation
import dataclasses
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer



import pandas as pd


from sklearn.model_selection import train_test_split

from dataclasses import  dataclass

@dataclass

class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts', 'train_data.csv')
    test_data_path :str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path :str = os.path.join('artifacts', 'data.csv')

# since we have configured the paths for the data files ..lets create class for data ingestion

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingest(self):
        logging.info('entered the data ingestion method')
        try :
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the data as dtaframe')
            
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path,index=False,header = True)

            logging.info('Raw data saved successfully')

        

            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)

            logging.info('data split successfully')

            train_set.to_csv(self.config.train_data_path,index=False,header = True)

            test_set.to_csv(self.config.test_data_path,index=False,header = True)

            logging.info('Train and test data saved successfully')


            return (
                self.config.train_data_path,
                self.config.test_data_path
                
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':

    obj = DataIngestion()

    train_data ,test_data =obj.initiate_data_ingest()

    data_transformation = DataTransformation()

    train_arr, test_arr , _ =data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
