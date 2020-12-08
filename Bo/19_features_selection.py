from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
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


def features_univariate_selection(file_directory, data):
	df = get_data(file_directory, data)
	# print(df.shape)
	# print(df[["text_id", "click_bait"]])

	X = df.iloc[:, 3:]
	y = df.iloc[:, 1:2]
	# print(X[:3])
	# print(y[:3])

	features_selector = SelectKBest(score_func=chi2, k=16)
	fit = features_selector.fit(X, y)
	df_scores = pd.DataFrame(fit.scores_)
	df_columns = pd.DataFrame(X.columns)
	# print(df_scores)
	# print(df_columns)

	featuresScores = pd.concat([df_columns, df_scores], axis=1)
	featuresScores.columns = ["feature", "chi2_score"]
	featuresScores.nlargest(16, 'chi2_score').set_index('feature').plot.barh()
	print(featuresScores.nlargest(16, 'chi2_score'))
	plt.title(label="k highest chi2 scores")
	plt.show()


def features_important(file_directory, data):
	df = get_data(file_directory, data)
	X = df.iloc[:, 3:]
	y = df.iloc[:, 1:2]
	y = np.squeeze(np.asarray(y))

	model = ExtraTreesClassifier()
	model.fit(X, y)
	df_scores = pd.DataFrame(model.feature_importances_)
	df_columns = pd.DataFrame(X.columns)
	scores = pd.concat([df_columns, df_scores], axis=1)
	scores.columns = ["feature", "score"]
	print(scores.nlargest(16, 'score'))

	feature_importance = pd.Series(model.feature_importances_, index=X.columns)
	feature_importance.nlargest(16).plot(kind='barh')
	plt.title(label="feature importance")
	plt.show()


def features_correlation(file_directory, data):
	df = get_data(file_directory, data)
	figure, axes = plt.subplots(figsize=(12, 12))
	sns.heatmap(df.drop(["post_text", 'text_id'], axis=1).corr(), annot=True, vmax=1, linewidths=.4, cmap='Reds')
	plt.title(label="feature correlation")
	plt.xticks(rotation=45)
	plt.show()


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# features_univariate_selection(file_directory, 1)
# features_important(file_directory, 1)
features_correlation(file_directory, 1)
