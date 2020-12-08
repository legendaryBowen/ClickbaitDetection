"""
split dataset to 8:2 ratio
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sqlite3
import random
import os

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()
sql = '''select text_id, post_text, truth_mean, truth_median, truth_mode, click_bait
				from main;'''

entries = cur.execute(sql)
entries = [list(e[:]) for e in entries]

df = pd.DataFrame(entries)

df.columns = ['text_id', 'post_text', 'truth_mean', 'truth_median', 'truth_mode', 'click_bait']

train, test = train_test_split(df, test_size=0.2, random_state=15)

print(train.shape)

for i in range(0, train.shape[0]):
	# print(train.iloc[i]['text_id'])
	cur.execute(
		"insert or ignore into main_train(text_id, post_text, truth_mean, truth_median, truth_mode, click_bait)"
		"values(?, ?, ?, ?, ?, ?)",
		(int(train.iloc[i]['text_id']), train.iloc[i]['post_text'], train.iloc[i]['truth_mean'],
		 train.iloc[i]['truth_median'], train.iloc[i]['truth_mode'], int(train.iloc[i]['click_bait'])))

for i in range(0, test.shape[0]):
	# print(test.iloc[i]['text_id'])
	cur.execute(
		"insert or ignore into main_test(text_id, post_text, truth_mean, truth_median, truth_mode, click_bait)"
		"values(?, ?, ?, ?, ?, ?)",
		(int(test.iloc[i]['text_id']), test.iloc[i]['post_text'], test.iloc[i]['truth_mean'],
		 test.iloc[i]['truth_median'], test.iloc[i]['truth_mode'], int(test.iloc[i]['click_bait'])))

db_connection.commit()
cur.close()
db_connection.close()
