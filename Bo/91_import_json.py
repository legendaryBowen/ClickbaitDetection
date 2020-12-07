import numpy as np
import pandas as pd
import sqlite3
import csv
import os
import json

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

text_id = []
LSTM_prob = []
GRU_prob = []

with open("test_b.json", 'r') as g:
	data = json.load(g)

for i in range(0, len(data)):
	text_id.append(data[i]['id'])
	LSTM_prob.append('{:.10f}'.format(data[i]['predicted_prob_LSTM']))
	GRU_prob.append('{:.10f}'.format(data[i]['predicted_prob_GRU']))

for i in range(0, len(data)):
	cur.execute("update main_test set predicted_prob_lstm = ?, predicted_prob_gru = ? where text_id = ?",
				(LSTM_prob[i], GRU_prob[i], int(text_id[i])))

db_connection.commit()
cur.close()
db_connection.close()
