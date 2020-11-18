#You will need to install these packages to see the data
import pandas as pd
import numpy as np


music_review_dataframe = pd.read_json("Digital_Music_5.json", lines=True)


#Look at the columns in our dataset
print(music_review_dataframe.columns)
print()

#How many reviews do we have?
print("total # of reviews:")
print(len(music_review_dataframe))
print()

#Lets take a look at some individual review entries
print(music_review_dataframe.reviewText[0])
print(music_review_dataframe.reviewText[100])
print(music_review_dataframe.reviewText[1000])
print()


#Compare a bad and a good review
bad_reviews_dataframe = music_review_dataframe[music_review_dataframe["overall"] <= 2.0]
good_reviews_dataframe = music_review_dataframe[music_review_dataframe["overall"] >= 4.0]

print(bad_reviews_dataframe.iloc[0, 8])
print(good_reviews_dataframe.iloc[0, 8])
print()


#How many reviews are there of each score?
print("Score of 1: ")
print(music_review_dataframe[music_review_dataframe['overall'] == 1.0].count().overall)
print("Score of 2: ")
print(music_review_dataframe[music_review_dataframe['overall'] == 2.0].count().overall)
print("Score of 3: ")
print(music_review_dataframe[music_review_dataframe['overall'] == 3.0].count().overall)
print("Score of 4: ")
print(music_review_dataframe[music_review_dataframe['overall'] == 4.0].count().overall)
print("Score of 5: ")
print(music_review_dataframe[music_review_dataframe['overall'] == 5.0].count().overall)

#Looks like we will need to do some data cleaning to remove stopwords and punctuation in order to perform something like Naive Bayes.

