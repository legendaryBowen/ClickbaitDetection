"""
This file:
1. calculates these columns in the “features" table:
"avg_word_length", "avg_word_length_normalization"

2. analyze the "avg_word_length" and plot the image
"""

from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os


def avg_word_length(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select text_id, post_text from main;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	length_avg = []
	for e in entries:
		result = regexp_tokenize(e[1], r"\w+")
		# print(result)
		sum_word_length = 0
		word_count = len(result)
		avg_word_length = 0
		for i in result:
			sum_word_length += len(i)
		# print(sum_word_length, word_count)
		if word_count == 0:
			pass
		else:
			avg_word_length = sum_word_length / word_count
		length_avg.append([e[0], avg_word_length])


	temp = np.array(length_avg)
	temp = temp[:, 1]
	temp = temp.astype(np.double)
	max = np.amax(temp)
	min = np.amin(temp)
	d = max - min

	for i in length_avg:
		text_id = i[0]
		avg_word_length = i[1]
		avg_word_length_normalization = (avg_word_length - min) / d
		cur.execute("update features set avg_word_length = ?, avg_word_length_norm = ?"
					" where text_id = ?", (avg_word_length, avg_word_length_normalization, text_id))

	db_connection.commit()
	cur.close()
	db_connection.close()


def avg_word_length_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select avg_word_length, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["avg_word_length", "click_bait"]
	df["avg_word_length"] = df["avg_word_length"].round()

	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("avg_word_length")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("AVG Word Length: AWL")
	pct_plot1.set_ylabel("% of Headlines with AWL")
	pct_plot1.set_title(label="'non-cb' and 'cb' distribution")

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel('AVG Word Length: AWL')
	pct_plot2.set_ylabel("Composition with AWL")
	pct_plot2.set_title(label="composition of 'non-cb' and 'cb' headlines")

	plt.show()

	print("deviation of 'avg_word_length' of clickbait tweet:")
	print(df_cb["avg_word_length"].std(axis=0))
	print("deviation of 'avg_word_length' of non-clickbait tweet:")
	print(df_ncb["avg_word_length"].std(axis=0))


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# avg_word_length(file_directory)
avg_word_length_analysis(file_directory)