import pandas as pd
import numpy as np
import texthero as hero
import timeit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

def clean_dataframe(df):
	df['clean_reviewText'] = hero.clean(df['reviewText'])
	return df

def train_svm(df):
    x_train, x_test, y_train, y_test = train_test_split(df['clean_reviewText'], df['overall'], test_size=0.2, random_state=42)
    print("Data read and train-test split successful")
    
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_train_counts = count_vect.fit_transform(x_train)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    print("Word count complete")
    
    clf = SVC()
    
    start = timeit.default_timer()
    clf.fit(x_train_tfidf, y_train)
    stop = timeit.default_timer()
    print("Model fitting complete, runtime = ",stop - start," seconds")
    
    x_test_transform = count_vect.transform(x_test)
    y_pred = clf.predict(x_test_transform)
    print("Model testing complete, metrics incoming")
    
    print("Confusion matrix : ", confusion_matrix(y_test,y_pred))
    print("Precision Score : ", precision_score(y_test,y_pred, average='micro'))
    print("Recall Score :" , recall_score(y_test, y_pred, average='micro') )
    

def main():
    print("Imports successful")
    
    mrdf = pd.read_json("Digital_Music_5.json", lines=True)
    mrdf = clean_dataframe(mrdf)
    print("Dataframe cleaned")
    #print(mrdf.reviewText[0])
    #print(mrdf.clean_reviewText[0])
    
    mrdf = mrdf.sample(frac = 0.3)
    print("Dataframe length:", len(mrdf))
    print("Dataset shrunk to 30% of original size")
    
    train_svm(mrdf)
    
    print("Main complete")

if __name__ == '__main__':
    main()