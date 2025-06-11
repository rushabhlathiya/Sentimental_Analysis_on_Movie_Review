import numpy as np
import pandas as pd
import pickle
import json
import os

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_X_test(filePath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filePath,header=None)
    except Exception as e:
        raise Exception(f"Error Loading Xtest form {filePath}:{e}")
# X_test = pd.read_csv('data/processed/X_test.csv',header=None)

def load_y_test(filePath:str)->pd.Series:
    try:
        return pd.read_csv(filePath)['label']
    except Exception as e:
        raise Exception(f"Error Loading ytest from {filePath}: {e}")
# y_test = pd.read_csv('data/processed/y_test.csv')['label']

def load_model(model_name:str):
    try:
        with open(model_name,'rb') as file:
            return pickle.load(file)        
    except Exception as e:
        raise Exception(f"Error loading Model {e}")
# model = pickle.load(open('model.pkl','rb'))

def evaluation_model(model,X_test:pd.DataFrame,y_test:pd.DataFrame)-> dict:
    try:
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        rec =recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)


        metrics_dict = {
            'acc':acc,
            'precision':pre,
            'recall': rec,
            'f1Score': f1
        }
        
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error Evaluation Model {e}")


def save_metrics(metrics_dict,filePath:str)-> None:
    try:
        with open(filePath,'w') as file:
            json.dump(metrics_dict,file,indent=4)
    except Exception as e:
        raise Exception(f"Error Saving Metrics to {filePath} : {e}")
    

def main():
    try:
        processed_filePath= 'data/processed'
        model_name ='model.pkl'
        metrics_name='metrics.json'
        
        X_test= load_X_test(os.path.join(processed_filePath,'X_test.csv'))
        y_test= load_y_test(os.path.join(processed_filePath,'y_test.csv'))
        
        model = load_model(model_name)
        
        metrics = evaluation_model(model,X_test,y_test)
        
        save_metrics(metrics,metrics_name)
    except Exception as e:
        raise Exception(f"An Error Occured {e}")
    
    
if __name__=='__main__':
    main()