"""
This files:
1. uses GridSearchCV to find the best regularization parameter.
2. uses 14 features and LR to predict click_bait.
"""

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions, plot_learning_curves
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
import pandas as pd
import numpy as np
import itertools
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
		sql = '''select m.post_text, m.click_bait, m.text_id, f.tweet_length_norm, f.avg_word_length_norm,
					f.stopword_count_norm, f.word_contraction_norm, f.starts_with_number, f.numOfPunctuationNorm,
					f.clickbaitWordsNorm, f.numOfNumericsNorm, f.sentiment_norm, f.stopword_pct_norm, f.noun_pct_norm,
					f.verb_pct_norm, f.preposition_pct_norm, f.qualifier_pct_norm, f.function_pct_norm,
					f.others_pct_norm
					from main m, features f where m.text_id=f.text_id;'''
	elif data == 2:
		sql = '''select m.post_text, m.click_bait, m.text_id, f.tweet_length_norm, f.avg_word_length_norm,
					f.stopword_count_norm, f.word_contraction_norm, f.starts_with_number, f.sentiment_norm, 
					f.stopword_pct_norm, f.noun_pct_norm, f.verb_pct_norm, f.preposition_pct_norm, 
					f.qualifier_pct_norm, f.function_pct_norm, f.others_pct_norm
					from main_2 m, features_2 f where m.text_id=f.text_id;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	df = pd.DataFrame(entries)
	if data == 1:
		df.columns = ["post_text", "click_bait", "text_id", "tweet_length_norm", "avg_word_length_norm",
					  "stopword_count_norm", "word_contraction_norm", "starts_with_number", "numOfPunctuationNorm",
					  "clickbaitWordsNorm", "numOfNumericsNorm", "sentiment_norm", "stopword_pct_norm", "noun_pct_norm",
					  "verb_pct_norm", "preposition_pct_norm", "qualifier_pct_norm", "function_pct_norm",
					  "others_pct_norm"]
	elif data == 2:
		df.columns = ["post_text", "click_bait", "text_id", "tweet_length_norm", "avg_word_length_norm",
					  "stopword_count_norm", "word_contraction_norm", "starts_with_number", "sentiment_norm",
					  "stopword_pct_norm", "noun_pct_norm", "verb_pct_norm", "preposition_pct_norm",
					  "qualifier_pct_norm", "function_pct_norm", "others_pct_norm"]
	return df


# def get_data_2(file_directory):
# 	db_path = file_directory + "\\click_baitX.sqlite"
# 	db_connection = sqlite3.connect(db_path)
# 	cur = db_connection.cursor()
#
# 	sql = '''select m.post_text, m.click_bait, m.text_id, f.tweet_length_norm, f.avg_word_length_norm,
# 				f.stopword_count_norm, f.word_contraction_norm, f.starts_with_number, f.sentiment_norm,
# 				f.stopword_pct_norm, f.noun_pct_norm, f.verb_pct_norm, f.preposition_pct_norm,
# 				f.qualifier_pct_norm, f.function_pct_norm, f.others_pct_norm
# 				from main_2 m, features_2 f where m.text_id=f.text_id;'''
#
# 	entries = cur.execute(sql)
# 	entries = [list(e[:]) for e in entries]
#
# 	df = pd.DataFrame(entries)
#
# 	df.columns = ["post_text", "click_bait", "text_id", "tweet_length_norm", "avg_word_length_norm",
# 				  "stopword_count_norm", "word_contraction_norm", "starts_with_number", "sentiment_norm",
# 				  "stopword_pct_norm", "noun_pct_norm", "verb_pct_norm", "preposition_pct_norm",
# 				  "qualifier_pct_norm", "function_pct_norm", "others_pct_norm"]
# 	return df


def get_report(model, X_train, X_test, y_train, y_test, threshold):
	model.fit(X_train, y_train)
	coefficients = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(np.transpose(model.coef_))], axis=1)
	coefficients.columns = ['feature', 'coefficient']
	print(coefficients.sort_values(by='coefficient', ascending=False))
	predict_decimal = model.predict_proba(X_test)
	predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
	predict_decimal = predict_decimal[:][:, 1:2]
	# print(X_test.iloc[[604], [1]], X_test.iloc[[604], [2]], X_test.iloc[[604], [3]], X_test.iloc[[604], [10]])
	# print(y_test[604])
	# print(predict_decimal[604])
	predict = np.int64(predict_decimal >= threshold)
	report = classification_report(y_test, predict)
	return report


def skfold_LR(file_directory, k, threshold, data):
	reports = []
	threshold = threshold
	df = get_data(file_directory, data)

	X = df.iloc[:, 3:]
	# X.drop(['sentiment_norm', 'starts_with_number', 'numOfNumericsNorm', 'verb_pct_norm',
	# 		'word_contraction_norm', 'others_pct_norm'], axis=1, inplace=True)
	y = df.iloc[:, 1:2]

	kf = StratifiedKFold(n_splits=k)
	for train_index, test_index in kf.split(X, y):
		X_train, X_test, y_train, y_test = X.iloc[list(train_index)], X.iloc[list(test_index)], \
										   y.iloc[list(train_index)], y.iloc[list(test_index)]
		y_train = np.squeeze(np.asarray(y_train))
		y_test = np.squeeze(np.asarray(y_test))

		"""
		find the best regularization parameter
		"""
		param_grid = [{'C': np.linspace(80, 100, 30)}]
		log_reg = LogisticRegression(max_iter=3000)
		grid_cv = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=3)
		grid_cv.fit(X_train, y_train)
		print('C =', grid_cv.best_params_['C'])
		print('best score:', grid_cv.best_score_)

		model = LogisticRegression(penalty='l2', C=grid_cv.best_params_['C'], max_iter=3000)
		reports.append(get_report(model, X_train, X_test, y_train, y_test, threshold))

	for report in reports:
		print(report)


def skfold_SVM(file_directory, k, kernel, threshold, data):
	reports = []
	threshold = threshold
	df = get_data(file_directory, data)

	X = df.iloc[:, 3:]
	# X.drop(['sentiment_norm', 'starts_with_number', 'numOfNumericsNorm', 'verb_pct_norm',
	# 		'word_contraction_norm', 'others_pct_norm'], axis=1, inplace=True)
	y = df.iloc[:, 1:2]

	kf = StratifiedKFold(n_splits=k)
	i = 0
	for train_index, test_index in kf.split(X, y):
		X_train, X_test, y_train, y_test = X.iloc[list(train_index)], X.iloc[list(test_index)], \
										   y.iloc[list(train_index)], y.iloc[list(test_index)]
		y_train = np.squeeze(np.asarray(y_train))
		y_test = np.squeeze(np.asarray(y_test))

		print(i)
		i += 1
		"""
		find the best regularization parameter
		"""
		param_grid = [{'C': np.linspace(80, 100, 30)}]
		svm_svc = svm.SVC(kernel=kernel, probability=True)
		grid_cv = GridSearchCV(svm_svc, param_grid, scoring='accuracy', cv=2)
		grid_cv.fit(X_train, y_train)
		print('C =', grid_cv.best_params_['C'])
		print('best score:', grid_cv.best_score_)

		model = svm.SVC(kernel=kernel, C=grid_cv.best_params_['C'], probability=True)
		reports.append(get_report(model, X_train, X_test, y_train, y_test, threshold))

	# # online  #
	# param_grid = {'C': [0.1, 1, 10, 100, 1000],
	# 			  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
	# 			  'kernel': [kernel]}
	# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
	# grid.fit(X_train, y_train)
	# print(grid.best_params_)
	# print(grid.best_estimator_)
	# grid_predictions = grid.predict(X_test)
	# reports.append(classification_report(y_test, grid_predictions))
	# # print(classification_report(y_test, grid_predictions))
	# #  end #

	for report in reports:
		print(report)


def RF(file_directory, depth_start, depth_end, threshold, data):
	df = get_data(file_directory, data)
	X = df.iloc[:, 3:]
	# X.drop(['sentiment_norm', 'starts_with_number', 'numOfNumericsNorm', 'verb_pct_norm',
	# 		'word_contraction_norm', 'others_pct_norm'], axis=1, inplace=True)
	y = df.iloc[:, 1:2]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
	y_train = np.squeeze(np.asarray(y_train))
	y_test = np.squeeze(np.asarray(y_test))

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
	cb_sum = []

	for i in range(depth_start, depth_end):
		r = random.randint(0, 100)
		model = RandomForestClassifier(max_depth=i, random_state=r, max_features='auto')
		model.fit(X_train, y_train)

		predict_decimal = model.predict_proba(X_test)

		predict_decimal = (predict_decimal - np.min(predict_decimal)) / (
				np.max(predict_decimal) - np.min(predict_decimal))
		predict_decimal = predict_decimal[:][:, 1:2]
		predict = np.int64(predict_decimal >= threshold)

		# predict = model.predict(X_test)
		acc = accuracy_score(y_test, predict)
		accuracy.append(acc)

		print(classification_report(y_test, predict))
		report = classification_report(y_test, predict, output_dict=True)
		depth.append(i)
		precision_0.append(report["0"]["precision"])
		recall_0.append(report["0"]["recall"])
		f1_0.append(report["0"]["f1-score"])
		precision_1.append(report["1"]["precision"])
		recall_1.append(report["1"]["recall"])
		f1_1.append(report["1"]["f1-score"])
		macro_precision.append(report['macro avg']['precision'])
		macro_recall.append(report['macro avg']['recall'])
		macro_f1.append(report['macro avg']['f1-score'])
		cb_sum.append(report["1"]["precision"] + report["1"]["recall"] + report["1"]["f1-score"])

	plt.figure(figsize=(8, 8))
	plt.plot(depth, precision_0, label="non_cb precision")
	plt.plot(depth, precision_1, label="cb precision")
	plt.plot(depth, macro_precision, label="macro avg precision")
	plt.title("Prediction Precision")
	plt.legend(loc="upper right")
	plt.xticks(depth, rotation=45)
	plt.xlabel("Depth")
	plt.ylabel("%")
	plt.show()

	plt.figure(figsize=(8, 8))
	plt.plot(depth, recall_0, label="non_cb recall")
	plt.plot(depth, recall_1, label="cb recall")
	plt.plot(depth, macro_recall, label="macro avg recall")
	plt.title("Prediction Recall")
	plt.legend(loc="upper right")
	plt.xticks(depth, rotation=45)
	plt.xlabel("Depth")
	plt.ylabel("%")
	plt.show()

	plt.figure(figsize=(8, 8))
	plt.plot(depth, accuracy, label="accuracy")
	plt.title("Prediction Accuracy")
	plt.legend(loc="upper right")
	plt.xticks(depth, rotation=45)
	plt.xlabel("Depth")
	plt.ylabel("%")
	plt.show()

	plt.figure(figsize=(8, 8))
	plt.plot(depth, f1_0, label="non_cb f1")
	plt.plot(depth, f1_1, label="cb f1")
	plt.plot(depth, macro_f1, label="macro avg f1")
	plt.title("Prediction F1")
	plt.legend(loc="upper right")
	plt.xticks(depth, rotation=45)
	plt.xlabel("Depth")
	plt.ylabel("%")
	plt.show()

	print(max(cb_sum), cb_sum.index(max(cb_sum)))


def hybrid(model1, weight1, model2, weight2, names, data, threshold):
	df = get_data(file_directory, data)
	X = df.iloc[:, 3:]
	# X.drop(['sentiment_norm', 'starts_with_number', 'numOfNumericsNorm', 'verb_pct_norm',
	# 		'word_contraction_norm', 'others_pct_norm'], axis=1, inplace=True)
	y = df.iloc[:, 1:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
	y_train = np.squeeze(np.asarray(y_train))
	y_test = np.squeeze(np.asarray(y_test))

	ensemble = VotingClassifier(estimators=[(names[0], model1), (names[1], model2)],
								voting='soft', weights=[weight1, weight2])
	ensemble.fit(X_train, y_train)
	predict_decimal = ensemble.predict_proba(X_test)
	predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
	predict_decimal = predict_decimal[:][:, 1:2]
	predict = np.int64(predict_decimal >= threshold)
	report = classification_report(y_test, predict)
	print(report)


def stacking(file_directory, data, threshold, kernel):
	pca = PCA(n_components=2)
	df = get_data(file_directory, data)
	X = df.iloc[:, 3:]
	# X.drop(['sentiment_norm', 'starts_with_number', 'numOfNumericsNorm', 'verb_pct_norm',
	# 		'word_contraction_norm', 'others_pct_norm'], axis=1, inplace=True)
	y = df.iloc[:, 1:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
	X_train = np.array(X_train)
	X_train2 = pca.fit_transform(X_train)
	X_test = np.array(X_test)
	X_test2 = pca.fit_transform(X_test)
	y_train = np.squeeze(np.asarray(y_train))
	y_test = np.squeeze(np.asarray(y_test))
	# print(type(X_train), type(X_test), type(y_train), type(y_test))

	clf1 = RandomForestClassifier(max_depth=30, n_estimators=10, random_state=42)
	clf2 = svm.SVC(kernel=kernel, probability=True)

	estimators = [('rf', clf1), ('svm', clf2)]
	sclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
	sclf.fit(X_train, y_train)
	predict_decimal = sclf.predict_proba(X_test)
	predict_decimal = (predict_decimal - np.min(predict_decimal)) / (np.max(predict_decimal) - np.min(predict_decimal))
	predict_decimal = predict_decimal[:][:, 1:2]
	predict = np.int64(predict_decimal >= threshold)
	report = classification_report(y_test, predict)
	print(report)

	label = ['RF', 'SVM', 'Stacking Classifier']
	clf_list = [clf1, clf2, sclf]
	fig = plt.figure(figsize=(10, 8))
	gs = gridspec.GridSpec(2, 2)
	grid = itertools.product([0, 1], repeat=2)

	clf_cv_mean = []
	clf_cv_std = []
	for clf, label, grd in zip(clf_list, label, grid):
		scores = cross_val_score(clf, X_train2, y_train, cv=3, scoring='accuracy')
		print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
		clf_cv_mean.append(scores.mean())
		clf_cv_std.append(scores.std())
		clf.fit(X_train2, y_train)
		ax = plt.subplot(gs[grd[0], grd[1]])
		fig = plot_decision_regions(X=X_train2, y=y_train, clf=clf, legend=2)
		plt.title(label)

	plt.show()
	plt.figure()
	(_, caps, _) = plt.errorbar(range(3), clf_cv_mean, yerr=clf_cv_std, c='blue', fmt='-o', capsize=5)
	for cap in caps:
		cap.set_markeredgewidth(1)
	plt.xticks(range(3), ['SVM', 'RF', 'Stacking'])
	plt.ylabel('Accuracy')
	plt.xlabel('Classifier')
	plt.title('Stacking Ensemble')
	plt.show()

	plt.figure()
	plot_learning_curves(X_train2, y_train, X_test2, y_test, sclf, print_model=False, style='ggplot')
	plt.show()


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

"""
with all features: (>0.32)
              precision    recall  f1-score   support

           0       0.87      0.86      0.87      2861
           1       0.63      0.63      0.63      1031

    accuracy                           0.80      3892
   macro avg       0.75      0.75      0.75      3892
weighted avg       0.80      0.80      0.80      3892
"""
skfold_LR(file_directory, 5, 0.33, 1)

"""
with all features: (>0.22), kernel='rbf'
              precision    recall  f1-score   support

           0       0.87      0.87      0.87      2861
           1       0.63      0.64      0.64      1031

    accuracy                           0.81      3892
   macro avg       0.75      0.75      0.75      3892
weighted avg       0.81      0.81      0.81      3892
"""
"""
with all features: (>0.35), kernel='linear'
              precision    recall  f1-score   support

           0       0.87      0.87      0.87      2861
           1       0.63      0.63      0.63      1031

    accuracy                           0.81      3892
   macro avg       0.75      0.75      0.75      3892
weighted avg       0.81      0.81      0.81      3892
"""
"""
with all features: (>0.22), kernel='poly'
              precision    recall  f1-score   support

           0       0.87      0.87      0.87      2861
           1       0.63      0.64      0.63      1031

    accuracy                           0.81      3892
   macro avg       0.75      0.75      0.75      3892
weighted avg       0.81      0.81      0.81      3892
"""
# skfold_SVM(file_directory, 3, 'rbf', 0.22, 1)  # 'rbf', 'linear', 'poly'


"""
random forest: (>0.3)
              precision    recall  f1-score   support

           0       0.88      0.84      0.86      2861
           1       0.61      0.68      0.64      1031

    accuracy                           0.80      3892
   macro avg       0.74      0.76      0.75      3892
weighted avg       0.81      0.80      0.80      3892
"""
# RF(file_directory, 2, 30, 0.5, 2)


# """hybrid - VotingClassifier"""
# model1 = LogisticRegression(multi_class='multinomial', random_state=15)
# model2 = RandomForestClassifier(max_depth=30, n_estimators=50, random_state=23)
# hybrid(model1, 2, model2, 3, ['LR', 'RF'], 1, 0.35)


"""
> 0.27 (svm: linear)
              precision    recall  f1-score   support

           0       0.87      0.88      0.87      2927
           1       0.62      0.60      0.61       965

    accuracy                           0.81      3892
   macro avg       0.74      0.74      0.74      3892
weighted avg       0.81      0.81      0.81      3892
"""
# stacking(file_directory, 1, 0.27, 'linear')
