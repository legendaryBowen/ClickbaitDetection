"""
This file imports "post_text" and "truth" from .json files to table "main".
"""
from random import randint
import jsonlines
import sqlite3
import os
import re

# import data from 1st dataset:
def first_dataset(file_directory):
	#  create database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	cur.executescript('''
			Drop table if exists main;
			create table main(
				text_id integer nnot null unique primary key,
				post_text text,
				click_bait integer,
				annotator1 real,
				annotator2 real,
				annotator3 real,
				annotator4 real,
				annotator5 real,
				truth_mean real,
				truth_median real,
				truth_mode real); 
	''')
	instance_file_path = file_directory + "\\instances.jsonl"
	truth_file_path = file_directory + "\\truth.jsonl"

	instance_file = open(instance_file_path, "r", encoding="utf-8")

	for e in jsonlines.Reader(instance_file):
		post_text = e["postText"][0]
		post_text = post_text.encode("ascii", "ignore")  # remove non-ascii character
		post_text = post_text.decode("utf-8")
		post_text = post_text.replace('\n', '').replace('\r', '')  # remove \n and \r in the string
		post_text = re.sub("(@\S+)", '', post_text)  # remove @mention
		post_text = re.sub("(#\S+)", '', post_text)  # remove #hashtag
		post_text = post_text.strip()  # remove whitespace
		post_text = " ".join(post_text.split())  # remove redundant space in the string
		text_id = int(e["id"])

		if post_text:
			cur.execute("insert or ignore into main(text_id, post_text) values(?, ?)", (text_id, post_text))


	# read truth file and add entries to database
	truth_file = open(truth_file_path, "r", encoding="utf8")
	for e in jsonlines.Reader(truth_file):
		text_id = int(e["id"])
		annotator1 = float(e["truthJudgments"][0])
		annotator2 = float(e["truthJudgments"][1])
		annotator3 = float(e["truthJudgments"][2])
		annotator4 = float(e["truthJudgments"][3])
		annotator5 = float(e["truthJudgments"][4])
		truth_mean = float(e["truthMean"])
		truth_median = float(e["truthMedian"])
		truth_mode = float(e["truthMode"])

		click_bait = 0
		if e["truthClass"] == "no-clickbait":
			pass
		elif e["truthClass"] == "clickbait":
			click_bait = 1

		sql = '''update main
					set annotator1 = ?, annotator2 = ?, annotator3 = ?, annotator4 = ?, annotator5 = ?,
						truth_mean = ?, truth_median = ?, truth_mode = ?, click_bait = ?
					where text_id = ?'''
		data = (annotator1, annotator2, annotator3, annotator4, annotator5, truth_mean, truth_median, truth_mode, click_bait, text_id)
		cur.execute(sql, data)

	db_connection.commit()
	cur.close()
	db_connection.close()

# import 10000 click bait headlines from the 2nd dataset
def second_dataset(file_directory):
	# file
	file_path = file_directory + "\\clickbait_data.txt"

	with open(file_path, encoding="utf-8") as f_in:
		headlines = [line.rstrip() for line in f_in]

	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	i = 1
	for headline in headlines:
		if i % 3 != 0:
			text_id = randint(900000000000000000, 990000000000000000)
			cur.execute("insert or ignore into main(text_id, post_text, click_bait) values(?, ?, ?)",
						(text_id, headline, 1))
		else:
			pass
		i += 1

	db_connection.commit()
	cur.close()
	db_connection.close()


# start from here
os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

first_dataset(file_directory)
# second_dataset(file_directory)
