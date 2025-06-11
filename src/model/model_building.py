import pandas as pd
import numpy as np
from sklearn.svm import SVC
import pickle
import yaml
import os

def load_X_train(filePath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filePath,header=None)
    except Exception as e:
        raise Exception(f"Error Loading Xtrain form {filePath}:{e}")
# X_train = pd.read_csv('data/processed/X_train.csv',header=None)

def load_y_train(filePath:str)->pd.Series:
    try:
        return pd.read_csv(filePath)['label']
    except Exception as e:
        raise Exception(f"Error Loading ytrain from {filePath}: {e}")
# y_train = pd.read_csv('data/processed/y_train.csv')['label']

def load_params(filePath:str)->float:
    try:
        with open(filePath,'r') as file:
            params =yaml.safe_load(file)
        return params['model_building']['C']
    except Exception as e:
        raise Exception(f"Error Loading Params from {filePath}: {e}")
# C = yaml.safe_load(open('params.yaml'))['model_building']['C']

def train_model(X_train:pd.DataFrame,y_train:pd.Series,C:float)->SVC:
    try:
        svm = SVC(C=C)
        svm.fit(X_train,y_train)
        return svm
    except Exception as e:
        raise Exception(f"Error Training Model : {e}")
# svm = SVC(C=C)
# svm.fit(X_train,y_train)

def save_model(model: SVC, filePath:str)->None:
    try:
        pickle.dump(model,open(filePath,'wb'))
    except Exception as e:
        raise Exception(f"Error Saving model to {filePath} : {e}")
# pickle.dump(svm,open("model.pkl","wb"))


def main():
    processed_filePath = 'data/processed'
    params_filePath ='params.yaml'
    model_name = "models/model.pkl"

    X_train = load_X_train(os.path.join(processed_filePath,'X_train.csv'))
    y_train = load_y_train(os.path.join(processed_filePath,'y_train.csv'))
    
    C= load_params(params_filePath)
    
    model = train_model(X_train,y_train,C)
    
    save_model(model,model_name)


if __name__=='__main__':
    main()