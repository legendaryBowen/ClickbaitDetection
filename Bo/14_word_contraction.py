"""
This file:
1. calculates these columns in the “features" table:
"word_contraction", "word_contraction_normalization"

2. analyze the "word_contraction" and plot the image
"""

from nltk.tokenize import TweetTokenizer
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import os


contraction_list = ["ain't", "aren't", "can't",  "could've", "couldn't", "couldn't've", "didn't", "doesn't", "don't", "hadn't", "hadn't've", "hasn't", "haven't", "he'd", "he'd've", "he'll", "he'll've", "he's", "how'd", "how'd'y", "how'll", "how's", "I'd", "I'd've", "I'll", "I'll've", "I'm", "I've", "isn't", "it'd", "it'd've", "it'll", "it'll've", "it's", "let's", "ma'am", "mayn't", "might've", "mightn't", "mightn't've", "must've", "mustn't", "mustn't've", "needn't", "needn't've", "o'clock", "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've", "she'd", "she'd've", "she'll", "she'll've", "she's", "should've", "shouldn't", "shouldn't've", "so've", "so's", "that'd", "that'd've", "that's", "there'd", "there'd've", "there's", "they'd", "they'd've", "they'll", "they'll've", "they're", "they've", "to've", "wasn't", "we'd", "we'd've", "we'll", "we'll've", "we're", "we've", "weren't", "what'll", "what'll've", "what're", "what's", "what've", "when's", "when've", "where'd", "where's", "where've", "who'll", "who'll've", "who's", "who've", "why's", "why've", "will've", "won't", "won't've", "would've", "wouldn't", "wouldn't've", "y'all", "y'all'd", "y'all'd've", "y'all're", "y'all've", "you'd", "you'd've", "you'll", "you'll've", "you're", "you've",
					"aint", "arent", "cant",  "couldve", "couldnt", "couldntve", "didnt", "doesnt", "dont", "hadnt", "hadntve", "hasnt", "havent", "hed", "hedve", "hell", "hellve", "hes", "howd", "howdy", "howll", "hows", "Id", "Idve", "Ill", "Illve", "Im", "Ive", "isnt", "itd", "itdve", "itll", "itllve", "its", "lets", "maam", "maynt", "mightve", "mightnt", "mightntve", "mustve", "mustnt", "mustntve", "neednt", "needntve", "oclock", "oughtnt", "oughtntve", "shant", "shant", "shantve", "shed", "shedve", "shell", "shellve", "shes", "shouldve", "shouldnt", "shouldntve", "sove", "sos", "thatd", "thatdve", "thats", "thered", "theredve", "theres", "theyd", "theydve", "theyll", "theyllve", "theyre", "theyve", "tove", "wasnt", "wed", "wedve", "well", "wellve", "were", "weve", "werent", "whatll", "whatllve", "whatre", "whats", "whatve", "whens", "whenve", "whered", "wheres", "whereve", "wholl", "whollve", "whos", "whove", "whys", "whyve", "willve", "wont", "wontve", "wouldve", "wouldnt", "wouldntve", "yall", "yalld", "yalldve", "yallre", "yallve", "youd", "youdve", "youll", "youllve", "youre", "youve"]


def word_contraction(file_directory, contraction_list):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select text_id, post_text, click_bait from main;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	text_list = []
	click_bait_list = []
	for e in entries:
		text_list.append(e[1])
		click_bait_list.append(int(e[2]))

	size = len(text_list)
	word_contraction_count = np.zeros(size, dtype=int)

	tknz = TweetTokenizer()

	i = 0
	for post_text in text_list:
		post_text_tokens_temp = []
		post_text_tokens = tknz.tokenize(post_text)

		for token in post_text_tokens:
			token = token.lower()
			post_text_tokens_temp.append(token)
		post_text_tokens = post_text_tokens_temp

		for token in post_text_tokens:
			if token in contraction_list:
				word_contraction_count[i] += 1
		i += 1

	word_contraction_count_normalization = np.zeros(size, dtype=np.double)
	max_count = np.amax(word_contraction_count)
	min_count = np.amin(word_contraction_count)
	d = max_count - min_count

	for i in range(0, size):
		word_contraction_count_normalization[i] = (word_contraction_count[i] - min_count) / d

	for i in range(0, 19463):
		cur.execute("update features set word_contraction = ?, word_contraction_norm = ? where text_id = ?",
					(int(word_contraction_count[i]), word_contraction_count_normalization[i], entries[i][0]))

	db_connection.commit()
	cur.close()
	db_connection.close()


def word_contraction_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select word_contraction, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["word_contraction", "click_bait"]

	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("word_contraction")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Word Contraction Count")
	pct_plot1.set_ylabel("% of Headlines with Word Contraction")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Word Contraction Count")
	pct_plot2.set_ylabel("Composition of Word Contraction")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")

	plt.show()


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

# word_contraction(file_directory, contraction_list)
word_contraction_analysis(file_directory)
