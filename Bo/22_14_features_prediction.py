"""
This files:
1. uses GridSearchCV to find the best regularization parameter.
2. uses 14 features and LR to predict click_bait.
The overall Precision is: 0.77, recall is 0.69, f1 is 0.71
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
import random
import os

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

#  database
db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()
# sql = '''select m.text_id, m.post_text, m.click_bait, f.tweet_length_norm, f.avg_word_length_norm,
# 			f.stopword_pct, f.stopword_count_norm, f.word_contraction_norm,
# 			f.noun_pct, f.verb_pct, f.preposition_pct, f.function_pct, f.qualifier_pct, f.others_pct,
# 			starts_with_number
# 			from main m, features f where m.text_id=f.text_id;'''
sql = '''select m.text_id, m.post_text, m.click_bait, f.tweet_length_norm, f.avg_word_length_norm,
			f.stopword_pct, f.stopword_count_norm, f.word_contraction_norm,
			f.noun_pct, f.verb_pct, f.preposition_pct, f.function_pct, f.qualifier_pct, f.others_pct,
			starts_with_number, f.numOfPunctuationNorm, f.clickbaitWordsNorm, f.numOfNumericsNorm,
			f.sentiment_norm
			from main m, features f where m.text_id=f.text_id;'''
entries = cur.execute(sql)
entries = [list(e[:]) for e in entries]

df = pd.DataFrame(entries)

df.columns = ["text_id", "post_text", "click_bait", "tweet_length_norm", "avg_word_length_norm",
			  "stopword_pct", "stopword_count_norm", "word_contraction_norm", "noun_pct",
			  "verb_pct", "preposition_pct", "function_pct", "qualifier_pct", "others_pct",
			  "starts_with_number", "numOfPunctuationNorm", "clickbaitWordsNorm", "numOfNumericsNorm",
			  "sentiment_norm"]

# collinearity analysis
# print(df.shape)
# figure, axes = plt.subplots(figsize=(12, 12))
# sns.heatmap(df.drop(['text_id'], axis=1).corr(), annot=True, vmax=1, linewidths=.5, cmap='Reds')
# plt.xticks(rotation=45)
# plt.show()


features_list = ["tweet_length_norm", "avg_word_length_norm", "stopword_pct", "stopword_count_norm",
				 "word_contraction_norm", "noun_pct", "verb_pct", "function_pct", "qualifier_pct",
				 "others_pct", "starts_with_number", "numOfPunctuationNorm", "clickbaitWordsNorm",
				 "numOfNumericsNorm", "sentiment_norm"]
results_list = ["click_bait"]
X = pd.DataFrame(df, columns=features_list)
# X.drop(["stopword_pct", "stopword_count_norm", "tweet_length_norm",
# 		"function_pct", "numOfNumericsNorm"], axis=1, inplace=True)
y = pd.DataFrame(df, columns=results_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
y_train = np.squeeze(np.asarray(y_train))
y_test = np.squeeze(np.asarray(y_test))

"""
find the best regularization parameter
"""
# param_grid = [{'C': np.linspace(90, 100, 20)}]
# log_reg = LogisticRegression(max_iter=3000)
# grid_cv = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=2)
#
# grid_cv.fit(X_train, y_train)
# print(grid_cv.best_params_)
# print(grid_cv.best_score_)

"""
with all features:
              precision    recall  f1-score   support

           0       0.84      0.94      0.89      4412
           1       0.70      0.44      0.54      1426

    accuracy                           0.82      5838
   macro avg       0.77      0.69      0.71      5838
weighted avg       0.80      0.82      0.80      5838
"""
# model = LogisticRegression(penalty="l2", C=91.05263157894737, max_iter=3000)
# model = model.fit(X_train, y_train)
# predict = model.predict(X_test)
# # predict = model.predict(X_train)
#
# print(classification_report(y_test, predict))
# # print(classification_report(y_train, predict))


"""
with all features:
              precision    recall  f1-score   support

           0       0.82      0.96      0.89      4412
           1       0.75      0.37      0.49      1426

    accuracy                           0.82      5838
   macro avg       0.79      0.66      0.69      5838
weighted avg       0.81      0.82      0.79      5838
"""
# model = svm.SVC(kernel='rbf')
# model.fit(X_train,y_train)
# predict = model.predict(X_test)
# # predict = model.score(X_test,y_test)
# print(classification_report(y_test, predict))

"""
with all features:
              precision    recall  f1-score   support

           0       0.83      0.96      0.89      4412
           1       0.74      0.39      0.51      1426

    accuracy                           0.82      5838
   macro avg       0.78      0.67      0.70      5838
weighted avg       0.81      0.82      0.80      5838
"""
# model = svm.SVC(kernel='linear')
# model.fit(X_train,y_train)
# predict = model.predict(X_test)
# print(classification_report(y_test, predict))


"""
with all features:
              precision    recall  f1-score   support

           0       0.82      0.96      0.89      4412
           1       0.76      0.35      0.48      1426

    accuracy                           0.81      5838
   macro avg       0.79      0.66      0.68      5838
weighted avg       0.81      0.81      0.79      5838
"""
# model = svm.SVC(kernel='poly')
# model.fit(X_train,y_train)
# predict = model.predict(X_test)
# print(classification_report(y_test, predict))

"""
with all features:
              precision    recall  f1-score   support

           0       0.77      0.98      0.86      4412
           1       0.62      0.11      0.19      1426

    accuracy                           0.77      5838
   macro avg       0.70      0.54      0.52      5838
weighted avg       0.74      0.77      0.70      5838

"""
# model = MultinomialNB(fit_prior=True)
# model.fit(X_train,y_train)
# predict = model.predict(X_test)
# print(classification_report(y_test, predict))



"""
random forest
"""
#
# depth = []
# accuracy = []
# precision_0 = []
# precision_1 = []
# recall_0 = []
# recall_1 = []
# f1_0 = []
# f1_1 = []
# macro_precision = []
# macro_recall = []
# macro_f1 = []
#
# for i in range(2, 35):
# 	r = random.randint(0, 100)
# 	model = RandomForestClassifier(max_depth=i, random_state=r)
# 	model.fit(X_train,y_train)
# 	predict = model.predict(X_test)
# 	acc = accuracy_score(y_test, predict)
# 	accuracy.append(acc)
# 	print(acc)
#
# 	print(classification_report(y_test, predict))
# 	report = classification_report(y_test, predict, output_dict=True)
# 	depth.append(i)
# 	precision_0.append(report["0"]["precision"])
# 	recall_0.append(report["0"]["recall"])
# 	f1_0.append(report["0"]["f1-score"])
# 	precision_1.append(report["1"]["precision"])
# 	recall_1.append(report["1"]["recall"])
# 	f1_1.append(report["1"]["f1-score"])
# 	macro_precision.append(report['macro avg']['precision'])
# 	macro_recall.append(report['macro avg']['recall'])
# 	macro_f1.append(report['macro avg']['f1-score'])
#
# plt.figure(figsize=(8,8))
# plt.plot(depth, precision_0, label="non_cb precision")
# plt.plot(depth, precision_1, label="cb precision")
# plt.plot(depth, macro_precision, label="macro avg precision")
# plt.title("Prediction Precision")
# plt.legend(loc="upper right")
# plt.xticks(depth, rotation=45)
# plt.xlabel("Depth")
# plt.ylabel("%")
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.plot(depth, recall_0, label="non_cb recall")
# plt.plot(depth, recall_1, label="cb recall")
# plt.plot(depth, macro_recall, label="macro avg recall")
# plt.title("Prediction Recall")
# plt.legend(loc="upper right")
# plt.xticks(depth, rotation=45)
# plt.xlabel("Depth")
# plt.ylabel("%")
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.plot(depth, accuracy, label="accuracy")
# plt.title("Prediction Accuracy")
# plt.legend(loc="upper right")
# plt.xticks(depth, rotation=45)
# plt.xlabel("Depth")
# plt.ylabel("%")
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.plot(depth, f1_0, label="non_cb f1")
# plt.plot(depth, f1_1, label="cb f1")
# plt.plot(depth, macro_f1, label="macro avg f1")
# plt.title("Prediction F1")
# plt.legend(loc="upper right")
# plt.xticks(depth, rotation=45)
# plt.xlabel("Depth")
# plt.ylabel("%")
# plt.show()
