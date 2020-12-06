import pandas as pd
import numpy as np
import texthero as hero
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score


def clean_dataframe(df):
	df['clean_reviewText'] = hero.clean(df['reviewText'])
	return df


def train_mnnb(df):
	####	
	####	Working on muddling with hyper parameters
	####

	# text_clf = Pipeline([('vect', CountVectorizer()),
 	#                     ('tfidf', TfidfTransformer()),
 	#                     ('clf', MultinomialNB())])

	# tuned_parameters = {
	#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
	#     'tfidf__use_idf': (True, False),
	#     'tfidf__norm': ('l1', 'l2'),
	#     'clf__alpha': [1, 1e-1, 1e-2]
	# }

	# clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring='f1')
	# clf.fit(x_train, y_train)

	# print(classification_report(y_test, clf.predict(x_test), digits=4))

	#Split train/test data 80/20
	x_train, x_test, y_train, y_test = train_test_split(df['clean_reviewText'], df['overall'], test_size=0.2, random_state=42)

	#Count words and embed them
	count_vect = CountVectorizer()
	tfidf_transformer = TfidfTransformer()
	x_train_counts = count_vect.fit_transform(x_train)
	x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	#Fit a vanilla MNNB classifier 
	clf = MultinomialNB()
	clf.fit(x_train_tfidf, y_train)

	#Test our model
	x_test_transform = count_vect.transform(x_test)

	y_pred = clf.predict(x_test_transform)
	print(confusion_matrix(y_test,y_pred))

	print("Precision Score : ",precision_score(y_test,y_pred, average='micro'))
	print("Recall Score :" , recall_score(y_test, y_pred, average='micro') )



def main():
	#Clean our data
	music_review_dataframe = pd.read_json("Digital_Music_5.json", lines=True)
	music_review_dataframe = clean_dataframe(music_review_dataframe)

	#Make sure our cleaning worked
	print(music_review_dataframe.reviewText[0])
	print(music_review_dataframe.reviewText[100])
	print(music_review_dataframe.reviewText[1000])

	print(music_review_dataframe.clean_reviewText[0])
	print(music_review_dataframe.clean_reviewText[100])
	print(music_review_dataframe.clean_reviewText[1000])

	#Test our model
	train_mnnb(music_review_dataframe)
	


if __name__ == '__main__':
	main()