"""
This file:
1. calculates these columns in the “features" table:
"numOfPunctuation", "numOfPunctuationNorm", "clickbaitWords",
"clickbaitWordsNorm", "numOfNumerics", "numOfNumericsNorm"
"""
import numpy as np
import pandas as pd
import sqlite3
import os


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

db_path = file_directory + "\\click_baitX.sqlite"
db_connection = sqlite3.connect(db_path)
cur = db_connection.cursor()

file_path = file_directory + "\\features_dataframe.csv"
data = pd.read_csv(file_path)

df = pd.DataFrame(data)
df.columns = ["text_id", "numOfPunctuation", "numOfPunctuationNorm",
			  "clickbaitWords", "clickbaitWordsNorm", "numOfNumerics", "numOfNumericsNorm"]


# --- Standardization --- #
df["numOfPunctuationNorm"] = (df["numOfPunctuationNorm"] - np.mean(df["numOfPunctuationNorm"])) / np.std(df["numOfPunctuationNorm"])
df["clickbaitWordsNorm"] = (df["clickbaitWordsNorm"] - np.mean(df["clickbaitWordsNorm"])) / np.std(df["clickbaitWordsNorm"])
df["numOfNumericsNorm"] = (df["numOfNumericsNorm"] - np.mean(df["numOfNumericsNorm"])) / np.std(df["numOfNumericsNorm"])


for i in range(0, df.shape[0]):
	cur.execute("update features_copy set numOfPunctuation = ?, numOfPunctuationNorm = ?,"
				"clickbaitWords= ?, clickbaitWordsNorm =?, numOfNumerics = ?, numOfNumericsNorm = ? "
				"where text_id = ?",
				(int(df["numOfPunctuation"][i]), df["numOfPunctuationNorm"][i], int(df["clickbaitWords"][i]),
				 df["clickbaitWordsNorm"][i], int(df["numOfNumerics"][i]), df["numOfNumericsNorm"][i],
				 int(df["text_id"][i]))
				)

db_connection.commit()
cur.close()
db_connection.close()
