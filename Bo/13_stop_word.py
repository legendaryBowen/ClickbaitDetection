"""
This file:
1. calculates these columns in the “features" table:
"stopword_count", "stopword_count_normalization",
"stopword_percentage", "stopword_percentage_normalization"

2. analyze the "stopword_count", "stopword_percentage" and plot the image
"""

from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os


def stop_word_ratio(file_directory):
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	# sql = '''select text_id, post_text from main;'''
	sql = '''SELECT m.text_id, m.post_text, f.tweet_length_count 
			from main m, features f where m.text_id=f.text_id'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	stop_words = set(stopwords.words("english"))
	stop_words_count = np.zeros((entries.shape[0]), dtype=np.int)
	stop_words_percentage = np.zeros((entries.shape[0]), dtype=np.double)

	lt = WordNetLemmatizer()

	i = 0
	for e in entries:
		lt_result = []
		result = regexp_tokenize(e[1], r"\w+")

		# Lemmatizing & lower the character
		for word in result:
			lt_result.append(lt.lemmatize(word))
		lt_result = np.array(lt_result)
		lt_result = [r.lower() for r in lt_result]

		for word in lt_result:
			if word in stop_words:
				stop_words_count[i] += 1

		tweet_length_count = int(e[2])
		if tweet_length_count == 0:
			stop_words_percentage[i] = 0
		else:
			stop_words_percentage[i] = stop_words_count[i] / tweet_length_count
		i += 1

	# --- normalization for stopword_count ---#
	# max = np.amax(stop_words_count)
	# min = np.amin(stop_words_count)
	# d = max - min
	# stop_words_count_normalization = np.zeros(len(stop_words_count), dtype=np.double)
	# for i in range(0, len(stop_words_count_normalization)):
	# 	stop_words_count_normalization[i] = (stop_words_count[i] - min) / d

	# --- Standardization --- #
	stop_words_count_normalization = (stop_words_count - np.mean(stop_words_count)) / np.std(stop_words_count)
	stop_words_percentage_normalization = (stop_words_percentage - np.mean(stop_words_percentage)) / np.std(stop_words_percentage)

	for i in range(0, len(entries)):
		c = int(stop_words_count[i])
		cur.execute("update features_copy set stopword_count = ?, stopword_pct = ?, stopword_count_norm = ?, "
					"stopword_pct_norm = ? where text_id = ?",
					(c, stop_words_percentage[i], stop_words_count_normalization[i], stop_words_percentage_normalization[i],
					 int(entries[i][0])))

	db_connection.commit()
	cur.close()
	db_connection.close()


def stop_word_ratio_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select stopword_count, stopword_pct_norm, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)
	df = pd.DataFrame(entries)

	df.columns = ["stopword_count", "stopword_pct", "click_bait"]
	df["stopword_pct"] = (df["stopword_pct"] * 10).round()
	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)
	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("stopword_count")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Stop Word Count: SWC")
	pct_plot1.set_ylabel("% of Headlines with SWC")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")
	plt.xticks(rotation=45)
	plt.show()

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Stop Word Count: SWC")
	pct_plot2.set_ylabel("Composition with SWC")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")
	plt.xticks(rotation=45)
	plt.show()

	df2 = df.groupby("stopword_pct")["click_bait"].value_counts().unstack()
	df2 = df2.divide(df2.sum(axis=0), axis=1)
	pct_plot3 = df2.plot(kind="bar", stacked=False)
	pct_plot3.set_xlabel("Stop Word Percentage: SWP")
	pct_plot3.set_ylabel("% of Headlines with SWP")
	pct_plot3.set_title(label="'non-cb' and 'cb' distribution")
	plt.xticks(rotation=45)
	plt.show()

	df2 = df2.divide(df2.sum(axis=1), axis=0)
	pct_plot4 = df2.plot(kind="bar", stacked=False)
	pct_plot4.set_xlabel("Stop Word Percentage: SWP")
	pct_plot4.set_ylabel("Composition with SWP")
	pct_plot4.set_title(label="composition of 'non-cb' and 'cb' headlines")
	plt.xticks(rotation=45)
	plt.show()

	print("standard deviation of 'stop_word_percentage' of click_bait",
		  np.std(df_cb["stopword_pct"]))
	print("standard deviation of 'stop_word_percentage' of non_click_bait",
		  np.std(df_ncb["stopword_pct"]))

	print("standard deviation of 'stop_word_count' of click_bait",
		  np.std(df_cb["stopword_count"]))
	print("standard deviation of 'stop_word_count' of non_click_bait",
		  np.std(df_ncb["stopword_count"]))


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# stop_word_ratio(file_directory)
stop_word_ratio_analysis(file_directory)
