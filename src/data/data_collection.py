import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
 
def load_data(filePath:str)-> pd.DataFrame:
    try:
        return pd.read_parquet(filePath)
    except Exception as e:
        raise Exception(f"Error Loading Data from {filePath}:{e}")
# df = pd.read_parquet(r'D:\DS\Project\Sentimental Analysis\dataset\train-00000-of-00001.parquet')

def load_params(filePath:str)-> float:
    try:
        with open(filePath,'r') as file:
            params = yaml.safe_load(file)
        return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error Loading Params form {filePath}:{e}")
# test_size =yaml.safe_load(open("params.yaml"))['data_collection']['test_size']

def split_data(df:pd.DataFrame,test_size:float)-> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(df,test_size=test_size,random_state=42, stratify=df['label'])
    except ValueError as e:
        raise ValueError(f"Error Spliting Dataset :{e}")
# train_data , test_data = train_test_split(df,test_size=test_size,random_state=42, stratify=df['label'])
# data_path =os.path.join('data','raw')
# os.makedirs(data_path)

def save_data(df:pd.DataFrame,filePath:str)->None:
    try:
        df.to_csv(filePath,index = False)
    except Exception as e:
        raise Exception(f"Error Saving data to {filePath} : {e}")
# train_data.to_csv(os.path.join(data_path,'train_data.csv'),index = False)
# test_data.to_csv(os.path.join(data_path,'test_data.csv'),index = False)


def main():
    try:
        data_filePath = r'dataset\train-00000-of-00001.parquet'
        params_filePath = 'params.yaml'
        raw_dataPath = os.path.join('data','raw')
        os.makedirs(raw_dataPath)
        
        df = load_data(data_filePath)
        test_size = load_params(params_filePath)
        train_data,test_data = split_data(df,test_size)
        save_data(train_data,os.path.join(raw_dataPath,'train_data.csv'))
        save_data(test_data,os.path.join(raw_dataPath,'test_data.csv'))
    except Exception as e:
        raise Exception(f"An error Occured: {e}")

if __name__=='__main__':
    main()
    
