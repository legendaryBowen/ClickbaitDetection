"""
This file
1. analyzes the n-gram based on frequency.
2. uses GridSearchCV to find the best regularization parameter.
3. uses n-gram, tf-idf, and Logistic Regression to make the prediction.
precision: 0.66, recall: 0.66, f1: 0.65
"""

from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from string import punctuation
from numpy import arange
import pandas as pd
import numpy as np
import sqlite3
import random
import os


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def get_data(file_directory, data):
	sql = None
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()
	if data == 1:
		sql = '''select post_text, click_bait from main;'''
	elif data == 2:
		sql = '''select post_text, click_bait from main_2;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	# tokenize and stemming
	RgTknz = RegexpTokenizer("[\w']+")
	lstemmer = LancasterStemmer()
	# pstemmer = PorterStemmer()

	i = 0
	for e in entries:
		temp_list = []
		tokens = RgTknz.tokenize(e[0])
		# print(tokens)
		# [pstemmer.stem(token) for token in tokens]
		for token in tokens:
			r = lstemmer.stem(token)
			temp_list.append(r)
		# print(len(temp_list))
		entries[i][0] = temp_list
		# print("\n")
		i += 1

	return entries


def n_gram_result_LR(C, X_train_tfidf, y_train, X_test_tfidf, y_test):
	model = LogisticRegression(penalty="l2", C=C, max_iter=3000)
	model.fit(X_train_tfidf, y_train)
	# predict = model.predict(X_test_tfidf)
	# # predict = model.predict(X_train_tfidf)
	# print(classification_report(y_test, predict))
	# # print(classification_report(y_train, predict))

	predict_decimal = model.predict_proba(X_test_tfidf)
	predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
	predict_decimal = predict_decimal[:][:, 1:2]
	predict = np.int64(predict_decimal >= 0.3)
	print(classification_report(y_test, predict))


def n_gram_prediction_LR(file_directory, min_df_start, min_df_end, max_df_start, max_df_end, min_ngram, max_ngram, data):
	entries = get_data(file_directory, data)

	best_parameter = [0, 0, 0, 0]
	X_train_tfidf = y_train = X_test_tfidf = y_test = None
	for i in range(min_df_start, min_df_end+1):
		min_df = i
		for j in arange(max_df_start, max_df_end+0.1, 0.1):
			j = round(j, 1)
			max_df = j

			# entries = get_data(file_directory)

			df = pd.DataFrame(data=entries)
			df.columns = ["post_text", "click_bait"]

			features_list = ["post_text"]
			results_list = ["click_bait"]

			# df_cb = df[(df["click_bait"] == 1)]
			# df_ncb = df[(df["click_bait"] == 0)]
			# df_ncb = df_ncb.sample(n=4700)
			#
			# df = pd.concat([df_cb, df_ncb], axis=0)
			# df = df.sample(frac=1)

			temp_list =[]
			for texts in df['post_text']:
				temp = ' '.join(t for t in texts if t not in punctuation)
				temp_list.append(temp)

			df['post_text'] = temp_list
			# print(df['post_text'][:2])
			# print(df['click_bait'][:2])
			X = pd.DataFrame(df, columns=features_list)
			y = pd.DataFrame(df, columns=results_list)
			# print(X[:2])
			# print(y[:2])
			# print(X.shape, y.shape)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
			y_train = np.squeeze(np.asarray(y_train))
			y_test = np.squeeze(np.asarray(y_test))

			tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(min_ngram, max_ngram))
			X_train_tfidf = tfidf.fit_transform(X_train['post_text'])
			X_test_tfidf = tfidf.transform(X_test['post_text'])

			param_grid = [{'C': np.linspace(90, 100, 20)}]
			log_reg = LogisticRegression(max_iter=3000)
			grid_cv = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=5, verbose=1)
			grid_cv.fit(X_train_tfidf, y_train)

			if grid_cv.best_score_ > best_parameter[3]:
				best_parameter[0] = min_df
				best_parameter[1] = max_df
				best_parameter[2] = grid_cv.best_params_
				best_parameter[3] = grid_cv.best_score_

			print(min_df, max_df, grid_cv.best_params_, grid_cv.best_score_)

	print("best parameter", best_parameter)
	n_gram_result_LR(best_parameter[3], X_train_tfidf, y_train, X_test_tfidf, y_test)


def n_gram_prediction_RF(file_directory, min_df_start, min_df_end, max_df_start, max_df_end, min_ngram, max_ngram, depth_start, depth_end, data):
	entries = get_data(file_directory, data)

	# X_train_tfidf = y_train = X_test_tfidf = y_test = None
	for i in range(min_df_start, min_df_end+1):
		min_df = i
		for j in arange(max_df_start, max_df_end+0.1, 0.1):
			j = round(j, 1)
			max_df = j

			# entries = get_data(file_directory)

			df = pd.DataFrame(data=entries)
			df.columns = ["post_text", "click_bait"]

			features_list = ["post_text"]
			results_list = ["click_bait"]

			# df_cb = df[(df["click_bait"] == 1)]
			# df_ncb = df[(df["click_bait"] == 0)]
			# df_ncb = df_ncb.sample(n=4700)
			#
			# df = pd.concat([df_cb, df_ncb], axis=0)
			# df = df.sample(frac=1)

			temp_list =[]
			for texts in df['post_text']:
				temp = ' '.join(t for t in texts if t not in punctuation)
				temp_list.append(temp)

			df['post_text'] = temp_list
			# print(df['post_text'][:2])
			# print(df['click_bait'][:2])
			X = pd.DataFrame(df, columns=features_list)
			y = pd.DataFrame(df, columns=results_list)
			# print(X[:2])
			# print(y[:2])
			# print(X.shape, y.shape)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
			y_train = np.squeeze(np.asarray(y_train))
			y_test = np.squeeze(np.asarray(y_test))

			tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(min_ngram, max_ngram))
			X_train_tfidf = tfidf.fit_transform(X_train['post_text'])
			X_test_tfidf = tfidf.transform(X_test['post_text'])

			depth = []
			accuracy = []
			precision_0 = []
			precision_1 = []
			recall_0 = []
			recall_1 = []
			f1_0 = []
			f1_1 = []
			macro_precision = []
			macro_recall = []
			macro_f1 = []

			for k in range(depth_start, depth_end):
				r = random.randint(0, 100)
				model = RandomForestClassifier(max_depth=k, random_state=r)
				model.fit(X_train_tfidf,y_train)
				predict = model.predict(X_test_tfidf)
				acc = accuracy_score(y_test, predict)
				accuracy.append(acc)
				print(acc)

				print(classification_report(y_test, predict))
				report = classification_report(y_test, predict, output_dict=True)
				# depth.append(i)
				# precision_0.append(report["0"]["precision"])
				# recall_0.append(report["0"]["recall"])
				# f1_0.append(report["0"]["f1-score"])
				# precision_1.append(report["1"]["precision"])
				# recall_1.append(report["1"]["recall"])
				# f1_1.append(report["1"]["f1-score"])
				# macro_precision.append(report['macro avg']['precision'])
				# macro_recall.append(report['macro avg']['recall'])
				# macro_f1.append(report['macro avg']['f1-score'])
			#
			# plt.figure(figsize=(8,8))
			# plt.plot(depth, precision_0, label="non_cb precision")
			# plt.plot(depth, precision_1, label="cb precision")
			# plt.plot(depth, macro_precision, label="macro avg precision")
			# plt.title("Prediction Precision: " + str(min_df) + " " + str(max_df) + " " + str(min_ngram) + " " + str(max_ngram))
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
			# plt.title("Prediction Recall: " + str(min_df) + " " + str(max_df) + " " + str(min_ngram) + " " + str(max_ngram))
			# plt.legend(loc="upper right")
			# plt.xticks(depth, rotation=45)
			# plt.xlabel("Depth")
			# plt.ylabel("%")
			# plt.show()
			#
			# plt.figure(figsize=(8,8))
			# plt.plot(depth, accuracy, label="accuracy")
			# plt.title("Prediction Accuracy: " + str(min_df) + " " + str(max_df) + " " + str(min_ngram) + " " + str(max_ngram))
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
			# plt.title("Prediction F1: " + str(min_df) + " " + str(max_df) + " " + str(min_ngram) + " " + str(max_ngram))
			# plt.legend(loc="upper right")
			# plt.xticks(depth, rotation=45)
			# plt.xlabel("Depth")
			# plt.ylabel("%")
			# plt.show()


def n_gram_analysis(file_directory, min_fq, top_n, data):
	entries = get_data(file_directory, data)

	long_list = []
	# for e in entries[24:29]:
	for e in entries:
		for token in e[0]:
			long_list.append(token)

	filterC = lambda w: len(w) < 3

	biGramFinder = BigramCollocationFinder.from_words(long_list)
	biGramFinder.apply_word_filter(filterC)
	biGramFinder.apply_freq_filter(min_fq)
	r = biGramFinder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
	print("total 2-gram ", "frequency >", min_fq, "is", len(biGramFinder.ngram_fd.items()))
	print("top", top_n, "2-gram with min_frequency", min_fq)
	print(r)
	print("\n")

	triGramFinder = TrigramCollocationFinder.from_words(long_list)
	triGramFinder.apply_word_filter(filterC)
	triGramFinder.apply_freq_filter(min_fq)
	r = triGramFinder.nbest(TrigramAssocMeasures.likelihood_ratio, top_n)
	print("total 3-gram ", "frequency >", min_fq, "is", len(triGramFinder.ngram_fd.items()))
	print("top", top_n, "3-gram with min_frequency:", min_fq)
	print(r)
	print("\n")



	quadGramFinder = QuadgramCollocationFinder.from_words(long_list)
	quadGramFinder.apply_word_filter(filterC)
	quadGramFinder.apply_freq_filter(min_fq)
	r = quadGramFinder.nbest(QuadgramAssocMeasures.likelihood_ratio, top_n)
	print("total 4-gram ", "frequency >", min_fq, "is", len(quadGramFinder.ngram_fd.items()))
	print("top", top_n, "4-gram with min_frequency:", min_fq)
	print(r)
	print("\n")

# start from here
os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()


n_gram_analysis(file_directory, 70, 10, 1)

"""
best parameter [5, 0.4, {'C': 90.52631578947368}, 0.7453926093575287]
              precision    recall  f1-score   support

           0       0.81      0.94      0.87      4408
           1       0.63      0.31      0.41      1430

    accuracy                           0.79      5838
   macro avg       0.72      0.63      0.64      5838
weighted avg       0.76      0.79      0.76      5838
"""
# n_gram_prediction_LR(file_directory, 4, 5, 0.5, 0.6, 2, 3, 2)

# n_gram_prediction_RF(file_directory, 4, 5, 0.5, 0.6, 2, 3, 3, 40, 2)
