# first neural network with keras tutorial
from numpy import loadtxt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import sqlite3
import pandas as pd
import numpy as np
import os

"""
              precision    recall  f1-score   support

         0.0       0.87      0.90      0.88      2957
         1.0       0.64      0.57      0.60       935

    accuracy                           0.82      3892
   macro avg       0.75      0.73      0.74      3892
weighted avg       0.81      0.82      0.81      3892
"""

def get_data(file_directory, data):
	sql = None
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()
	if data == 1:
		sql = '''select f.text_id, f.click_bait, f.tweet_length_norm, f.avg_word_length_norm, f.stopword_count_norm, 
					f.word_contraction_norm, f.starts_with_number, f.numOfPunctuationNorm, f.clickbaitWordsNorm, 
					f.numOfNumericsNorm, f.sentiment_norm, f.stopword_pct_norm, f.noun_pct_norm, f.verb_pct_norm, 
					f.preposition_pct_norm, f.qualifier_pct_norm, f.function_pct_norm, f.others_pct_norm
					from features f ;'''
	elif data == 2:
		sql = '''select f.text_id, f.click_bait, f.tweet_length_norm, f.avg_word_length_norm, f.stopword_count_norm, 
					f.word_contraction_norm, f.starts_with_number, f.sentiment_norm, f.stopword_pct_norm, 
					f.noun_pct_norm, f.verb_pct_norm, f.preposition_pct_norm, f.qualifier_pct_norm, f.function_pct_norm, 
					f.others_pct_norm
					from features_2 f ;'''

	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	return entries


def nn_prediction(file_directory, epoch, data):
	entries = get_data(file_directory, data)
	entries = np.array(entries)
	X = entries[:, 2:]
	y = entries[:, 1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

	model = Sequential()
	dim = 0
	if data == 1:
		dim = 16
	elif data == 2:
		dim = 13

	model.add(Dense(12, input_dim=dim, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, epochs=epoch, batch_size=10)

	predict_decimal = model.predict(X_test)
	predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
	predict = np.int64(predict_decimal >= 0.35)
	print(classification_report(y_test, predict))


# start from here
os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

"""
> 0.3
              precision    recall  f1-score   support

         0.0       0.87      0.88      0.88      2957
         1.0       0.62      0.59      0.60       935

    accuracy                           0.81      3892
   macro avg       0.74      0.74      0.74      3892
weighted avg       0.81      0.81      0.81      3892
"""
nn_prediction(file_directory, 50, 1)
