from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, recall_score, f1_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import sqlite3
import os


test_count = 1

# read text from database
file_directory = os.getcwd()
db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

sql = '''select post_text, click_bait from main'''
entries = cur.execute(sql)
entries = [list(e[0:2]) for e in entries]
entries = np.array(entries)

result_accuracy = []
result_roc_auc = []
result_precision = []
result_recall = []
result_f1 = []
result_mse = []

for i in range(0, test_count):
	r = np.random.randint(1, 100)

	X = [e[0] for e in entries]
	y = [e[1] for e in entries]
	X = np.array(X)		# np.array, str, length=19470
	y = np.array(y).astype(int)  # np.array, int, length=19470


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r)
	# print(type(X_test), type(X_test[0]), type(y_train), type(y_train[0]))


	count_vec = CountVectorizer()
	X_train_matrix = count_vec.fit_transform(X_train)  # class 'scipy.sparse.csr.csr_matrix', shape: 14602*19800
	# print(type(X_matrix), X_matrix.shape)

	clf = MultinomialNB(fit_prior=True)
	clf.fit(X_train_matrix, y_train)

	# print(count_vec.get_feature_names())
	# print(clf.intercept_)
	# print(clf.coef_)

	X_test_matrix = count_vec.transform(X_test)

	print(X_test_matrix.shape)
	predicted = clf.predict(X_test_matrix)

	print(classification_report(y_test, predicted))

	# accuracy = accuracy_score(y_test, predicted)
	# result_accuracy.append(accuracy)
	# precision = precision_score(y_test, predicted, average="binary", pos_label=1)
	# result_precision.append(precision)
	# recall = recall_score(y_test, predicted, average="binary", pos_label=1)
	# result_recall.append(recall)
	# roc_auc = roc_auc_score(y_test, predicted)
	# result_roc_auc.append(roc_auc)
	# f1 = f1_score(y_test, predicted, average="binary", pos_label=1)
	# result_f1.append(f1)
	# mse = mean_squared_error(y_test, predicted)
	# result_mse.append(mse)


# print("test count: ", test_count)
#
# # print("accuracy: ", result_accuracy)
# print("accuracy mean: ", np.mean(result_accuracy))
#
# # print("precision: ", result_precision)
# print("precision mean: ", np.mean(result_precision))
#
# # print("recall: ", result_recall)
# print("recall mean: ", np.mean(result_recall))
#
# # print("roc_auc: ", result_roc_auc)
# print("roc_auc mean: ", np.mean(result_roc_auc))
#
# # print("f1: ", result_f1)
# print("f1 mean: ", np.mean(result_f1))
#
# # print("mse: ", result_mse)
# print("mse mean: ", np.mean(result_mse))
