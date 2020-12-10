import pandas as pd
import texthero as hero
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline, Pipeline

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
    
    param_grid = {'svc__C': [0.3, 0.7, 1.1], 'svc__gamma': [0.1,0.3,0.5], 'svc__kernel': ['rbf']}
    pipeline = make_pipeline(count_vect, SVC())
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    print("Parameter tuning complete")
    print("Best parameters : ", grid.best_params_)
    
    clf = SVC(C = grid.best_params_["svc__C"], gamma = grid.best_params_["svc__gamma"])
    clf.fit(x_train_tfidf, y_train)
    x_test_transform = count_vect.transform(x_test)
    y_pred = clf.predict(x_test_transform)
    print("Model fitting/testing complete, metrics incoming")
    print("Confusion matrix : ", confusion_matrix(y_test,y_pred))
    print("Precision Score : ", precision_score(y_test,y_pred, average='weighted'))
    print("Recall Score :" , recall_score(y_test, y_pred, average='weighted'))
    print("Accuracy Score : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average='weighted'))
    

def main():
    print("Imports successful")
    
    mrdf = pd.read_json("Pet_Supplies_5.json", lines=True)
    mrdf = mrdf.sample(frac = 0.02)
    print("Dataframe length:", len(mrdf))
    print("Dataset shrunk to 2% of original size")
    
    mrdf = clean_dataframe(mrdf)
    print("Dataframe cleaned")
    
    train_svm(mrdf)
    
    print("Main complete")

if __name__ == '__main__':
    main()