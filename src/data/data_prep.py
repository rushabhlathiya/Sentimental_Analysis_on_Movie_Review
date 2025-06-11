import numpy as np
import pandas as pd
import os 
import string
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import pickle


def load_data(filePath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filePath)
    except Exception as e:
        raise(f"Error Loading Data from {filePath} : {e}")
# train_data = pd.read_csv("data/raw/train_data.csv")
# test_data = pd.read_csv("data/raw/test_data.csv")


nltk.download('stopwords')
nltk.download('wordnet')
stopWords = stopwords.words('english')
wnl = WordNetLemmatizer()
translator = str.maketrans("","",string.punctuation)





def tokenaization_stopword(text):
    try:
        text = text.split()
        text= [x for x in text if x not in stopWords]
        text = [wnl.lemmatize(x,pos='v') for x in text]
        text = " ".join(text)
        return text
    except Exception as e:
        raise Exception(f"Error Lemmatizing Data {e}")

def clean_text(text):
    try:
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.translate(translator)
        text = re.sub(r'\s+'," ",text)
        return text
    except Exception as e:
        raise Exception(f"Error Cleaning Data {e}")
    
# data_path =os.path.join('data','processed')

# os.makedirs(data_path)
def save_X_data(df:np.array,filePath:str)->None:
    try:
        pd.DataFrame(df).to_csv(filePath,index=False)
    except Exception as e:
        raise Exception(f"Error Saving X :{e}")

def save_y_data(df,filePath:str)->None:
    try:
        pd.DataFrame(df).to_csv(filePath,index=False)
    except Exception as e:
        raise Exception(f"Error Saving y :{e}")
        
# pd.DataFrame(X_train).to_csv(os.path.join(data_path,'X_train.csv'),index=False,header=False)
# pd.DataFrame(X_test).to_csv(os.path.join(data_path,'X_test.csv'),index=False,header=False)
# pd.DataFrame(y_train).to_csv(os.path.join(data_path,'y_train.csv'),index=False)
# pd.DataFrame(y_test).to_csv(os.path.join(data_path,'y_test.csv'),index=False)

def save_object(obj:object,filePath:str)-> None:
    try:
        pickle.dump(obj,open(filePath, 'wb'))
    except Exception as e:
        raise Exception(f"Error Saving Object : {e}")
# pickle.dump(tfidf,open('tfidf.pkl', 'wb'))
# pickle.dump(w2v,open('w2v.pkl', 'wb'))





def main():
    try:
        raw_filePath = 'data/raw'
        processed_filePath = os.path.join('data','processed')
        os.makedirs(processed_filePath)
        
        train_data = load_data(os.path.join(raw_filePath,"train_data.csv"))
        test_data = load_data(os.path.join(raw_filePath,"test_data.csv"))
        
        train_data['cleanText'] = train_data['text'].apply(clean_text)
        test_data['cleanText'] = test_data['text'].apply(clean_text)


        train_data['tokenaizeText'] = train_data['cleanText'].apply(tokenaization_stopword)
        test_data['tokenaizeText'] = test_data['cleanText'].apply(tokenaization_stopword)


        tfidf = TfidfVectorizer(ngram_range =(1,2),max_features = 10000)
        X_train_tfidf=tfidf.fit_transform(train_data['tokenaizeText'])
        X_test_tfidf=tfidf.transform(test_data['tokenaizeText'])

        y_train = train_data['label']
        y_test = test_data['label']
        
        save_X_data(X_train_tfidf.toarray(),os.path.join(processed_filePath,'X_train_tfidf.csv'))
        save_X_data(X_test_tfidf.toarray(),os.path.join(processed_filePath,'X_test_tfidf.csv'))
        save_y_data(y_train,os.path.join(processed_filePath,'y_train.csv'))
        save_y_data(y_test,os.path.join(processed_filePath,'y_test.csv'))
        
    except Exception as e:
        raise Exception(f"An Error Occured {e}")
    

if __name__=='__main__':
    main()