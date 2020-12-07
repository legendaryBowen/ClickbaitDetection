"""
This file:
1. calculates these columns in the “features" table:
"tweet_length_count", "tweet_length_normalization"
2. analyze the "tweet_length_count" and plot the image
"""

from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def headline_length(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select text_id, post_text, click_bait from main;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	lt = WordNetLemmatizer()
	length_count = []
	for e in entries:
		lt_result = []
		result = regexp_tokenize(e[1], r"\w+")

		# Lemmatizing and lower the character
		for word in result:
			lt_result.append(lt.lemmatize(word))
		lt_result = np.array(lt_result)
		lt_result = [r.lower() for r in lt_result]

		length_count.append([e[0], len(lt_result), e[2]])


	temp = np.array(length_count)
	temp = temp[:, 1]
	temp = temp.astype(np.int)
	# --- Normalization --- #
	# min = np.amin(temp)
	# max = np.amax(temp)
	# d = max - min
	#
	#
	# for e in length_count:
	# 	text_id = e[0]
	# 	tweet_length_count = e[1]
	# 	tweet_length_normalization = (e[1] - min) / d
	# 	click_bait = e[2]
	#
	# 	cur.execute(
	# 		"insert or ignore into features_copy(text_id, tweet_length_count, tweet_length_norm, click_bait)"
	# 		"values(?, ?, ?, ?)", (text_id, tweet_length_count, tweet_length_normalization, click_bait))


	# --- Standardization --- #
	temp = temp.reshape(-1, 1)
	# print(temp[:4])
	sc = StandardScaler()
	tweet_length_normalization_list = sc.fit_transform((temp))
	# print(tweet_length_normalization_list[:4])
	tweet_length_normalization_list = tweet_length_normalization_list.reshape(1, -1)
	tweet_length_normalization_list = tweet_length_normalization_list[0]
	# print(tweet_length_normalization_list[:4])

	i = 0
	for e in length_count:
		text_id = e[0]
		tweet_length_count = e[1]
		tweet_length_normalization = tweet_length_normalization_list[i]
		click_bait = e[2]
		i += 1
		cur.execute(
			"insert or ignore into features_copy(text_id, tweet_length_count, tweet_length_norm, click_bait)"
			"values(?, ?, ?, ?)", (text_id, tweet_length_count, tweet_length_normalization, click_bait))



	db_connection.commit()
	cur.close()
	db_connection.close()


def headline_length_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select tweet_length_count, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["tweet_length_count", "click_bait"]

	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("tweet_length_count")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Tweet Length: TL")
	pct_plot1.set_ylabel("% of Headlines of Length TL")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")
	plt.xticks(rotation=45)
	plt.show()

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Tweet Length: TL")
	pct_plot2.set_ylabel("Composition with TL")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")
	plt.xticks(rotation=45)
	plt.show()

	print("deviation of 'tweet length' of clickbait tweet:")
	print(df_cb["tweet_length_count"].std(axis=0))
	print("deviation of 'tweet length' of non-clickbait tweet:")
	print(df_ncb["tweet_length_count"].std(axis=0))


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# headline_length(file_directory)
headline_length_analysis(file_directory)
