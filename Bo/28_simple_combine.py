from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import sqlite3
import random
import os

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
np.set_printoptions(suppress=True)

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

sql = '''select text_id, predicted_prob_lr, predicted_prob_rf, click_bait
				from main_test;'''

entries = cur.execute(sql)
entries = [list(e[:]) for e in entries]
entries = np.array(entries)

truth = []
for e in entries:
	truth.append(int(e[3]))
print(truth[:10])


# print(entries.shape)
s = 0
i = 0
index = []
for e in entries:
	if abs(e[1]-e[2]) >= 0 and abs(e[1]-e[2]) <= 0.1:
		s += 1
		index.append(i)
	i += 1

lr_list = []
lr_truth_list = []
rf_list = []
rf_truth_list = []
for id in index:
	lr_list.append(entries[id][1])
	lr_truth_list.append(int(entries[id][3]))

print("origial lr_list", lr_list[:5])
print(lr_truth_list[:5])

lr_list = np.array(lr_list)
lr_list = np.int64(lr_list >= 0.4)

print("after threshold lr_list", lr_list[:5])




print(classification_report(lr_truth_list, lr_list))