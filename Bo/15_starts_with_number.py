"""
This file:
1. calculates these columns in the “features" table:
"starts_with_number"

2. analyze the "starts_with_number" and plot the image
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os
import re


def regexp(expr, item):
	reg = re.compile(expr)
	return reg.search(item) is not None


def starts_with_number(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	db_connection.create_function("REGEXP", 2, regexp)
	cur = db_connection.cursor()

	sql = '''select text_id from main where post_text REGEXP '^[0-9]+'; '''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	cur.execute("update features_copy set starts_with_number = 0")

	# for e in entries:
	# 	cur.execute("update features_copy set starts_with_number = 1 where text_id = ?", (e[0],))

	db_connection.commit()
	cur.close()
	db_connection.close()


def starts_with_number_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select starts_with_number, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["starts_with_number", "click_bait"]

	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("starts_with_number")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Starts with Number: No/Yes")
	pct_plot1.set_ylabel("% of Headlines of Staring with Number")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Starts with Number: No/Yes")
	pct_plot2.set_ylabel("Composition of Starting with Number")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")

	plt.show()

	print("deviation of 'starts_with_number' of clickbait tweet:")
	print(df_cb["starts_with_number"].std(axis=0))
	print("deviation of 'starts_with_number' of non-clickbait tweet:")
	print(df_ncb["starts_with_number"].std(axis=0))


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# starts_with_number(file_directory)
starts_with_number_analysis(file_directory)
