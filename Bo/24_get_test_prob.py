from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
import pandas as pd
import numpy as np
import sqlite3
import random
import os

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
np.set_printoptions(suppress=True)

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

sql_train = '''select text_id, tweet_length_norm, avg_word_length_norm,
				stopword_count_norm, word_contraction_norm, starts_with_number, numOfPunctuationNorm,
				clickbaitWordsNorm, numOfNumericsNorm, sentiment_norm, stopword_pct_norm, noun_pct_norm,
				verb_pct_norm, preposition_pct_norm, qualifier_pct_norm, function_pct_norm,
				others_pct_norm, click_bait
				from features_train;'''

entries_train = cur.execute(sql_train)
entries_train = [list(e[:]) for e in entries_train]

df_train = pd.DataFrame(entries_train)

df_train.columns = ["text_id", "tweet_length_norm", "avg_word_length_norm",
					"stopword_count_norm", "word_contraction_norm", "starts_with_number", "numOfPunctuationNorm",
					"clickbaitWordsNorm", "numOfNumericsNorm", "sentiment_norm", "stopword_pct_norm", "noun_pct_norm",
					"verb_pct_norm", "preposition_pct_norm", "qualifier_pct_norm", "function_pct_norm",
					"others_pct_norm", "click_bait", ]

X_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:, -1]
y_train = np.squeeze(np.asarray(y_train))
# print(X_train.shape)
# print(y_train.shape)

sql_test = '''select text_id, tweet_length_norm, avg_word_length_norm,
				stopword_count_norm, word_contraction_norm, starts_with_number, numOfPunctuationNorm,
				clickbaitWordsNorm, numOfNumericsNorm, sentiment_norm, stopword_pct_norm, noun_pct_norm,
				verb_pct_norm, preposition_pct_norm, qualifier_pct_norm, function_pct_norm,
				others_pct_norm, click_bait
				from features_test;'''

entries_test = cur.execute(sql_test)
entries_test = [list(e[:]) for e in entries_test]

df_test = pd.DataFrame(entries_test)

df_test.columns = ["text_id", "tweet_length_norm", "avg_word_length_norm",
				   "stopword_count_norm", "word_contraction_norm", "starts_with_number", "numOfPunctuationNorm",
				   "clickbaitWordsNorm", "numOfNumericsNorm", "sentiment_norm", "stopword_pct_norm", "noun_pct_norm",
				   "verb_pct_norm", "preposition_pct_norm", "qualifier_pct_norm", "function_pct_norm",
				   "others_pct_norm", "click_bait", ]

X_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:, -1]
y_test = np.squeeze(np.asarray(y_test))
# print(X_test.shape)
# print(y_test.shape)

"""
find the best regularization parameter
"""
# param_grid = [{'C': np.linspace(80, 100, 30)}]
# log_reg = LogisticRegression(max_iter=3000)
# grid_cv = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=2)
# grid_cv.fit(X_train, y_train)
# # print('C =', grid_cv.best_params_['C'])
# # print('best score:', grid_cv.best_score_)
#
# model = LogisticRegression(penalty='l2', C=grid_cv.best_params_['C'], max_iter=3000)

model = RandomForestClassifier(max_depth=30, random_state=15, max_features='auto')
model.fit(X_train, y_train)
predict_decimal = model.predict_proba(X_test)

predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
predict_decimal = predict_decimal[:][:, 1:2]
predict = np.int64(predict_decimal >= 0.38)
print(classification_report(y_test, predict))

for i in range(0, X_test.shape[0]):
	cur.execute("update main_test set predicted_prob_rf = ? where text_id = ?",
				(predict_decimal[i][0], int(df_test['text_id'][i])))

db_connection.commit()
cur.close()
db_connection.close()
