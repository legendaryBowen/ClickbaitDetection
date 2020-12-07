from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import string
import os

from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')


def get_data(file_directory):
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select post_text, text_id from main;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	for i in range(0, len(entries)):
		entries[i][0] = entries[i][0].translate(str.maketrans('', '', string.punctuation))

	return entries


def get_sentiment(file_directory):
	sentiment_list = []
	sentiment_norm_list = []
	entries = get_data(file_directory)

	df = pd.DataFrame(entries)
	df.columns = ["post_text", "text_id"]
	texts = df["post_text"]
	text_id = df["text_id"]

	for text in texts:
		result = nlp.annotate(text, properties={
			'annotators': 'sentiment',
			'outputFormat': 'json', 'timeout': 10000})
		for s in result["sentences"]:
			# print("{} (Sentiment Value) {} (Sentiment)".format(
			# s["sentimentValue"], s["sentiment"]))
			sentiment_list.append(int(s["sentimentValue"]))

	# --- normalization for word_contraction_count ---#
	# max_sentiment = max(sentiment_list)
	# min_sentiment = min(sentiment_list)
	# # print(sentiment_list)
	# for sentiment in sentiment_list:
	# 	sentiment_norm = (sentiment - min_sentiment) / (max_sentiment - min_sentiment)
	# 	sentiment_norm_list.append(sentiment_norm)

	# --- Standardization --- #
	sentiment_norm_list = (sentiment_list - np.mean(sentiment_list)) / np.std(sentiment_list)

	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	for i in range(0, len(sentiment_list)):
		cur.execute("update features_copy set sentiment_value = ?, sentiment_norm = ?"
					" where text_id = ?", (sentiment_list[i], sentiment_norm_list[i], int(text_id[i])))

	db_connection.commit()
	cur.close()
	db_connection.close()


def sentiment_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select sentiment_value, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["sentiment_value", "click_bait"]
	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("sentiment_value")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Tweet Length: TL")
	pct_plot1.set_ylabel("% of Headlines of Length TL")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Tweet Length: TL")
	pct_plot2.set_ylabel("Composition with TL")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")

	plt.show()
	pass


# start from here
os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# get_sentiment(file_directory)
sentiment_analysis(file_directory)
