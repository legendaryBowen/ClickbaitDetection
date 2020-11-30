"""
The PoS has a total of 35 kinds of tags which are divided into 6 features.

This file:
1. calculates these columns in the “features" table:
"noun_pct", "verb_pct", "preposition_pct", "qualifier_pct", "function_pct", "others_pct"

2. analyze above features and plot the image
"""

from nltk.tokenize import regexp_tokenize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import nltk
import os


def pos(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select m.text_id, m.post_text, m.click_bait, f.tweet_length_count
	 			from main m, features f where m.text_id == f.text_id;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	text_id_list = []
	post_test_list = []
	click_bait_list = []
	tweet_length_count_list = []
	pos_count = np.zeros((19463, 35), dtype=np.int)
	pos_count_percentage = np.zeros((19463, 35), dtype=np.double)
	pos_item_list = ["CC", "CD", "DT", "EX", "FW",
					 "IN", "JJ", "JJR", "JJS", "LS",
					 "MD", "NN", "NNS", "NNP", "NNPS",
					 "PDT", "POS", "PRP", "PRP$", "RB",
					 "RBR", "RBS", "RP", "TO", "UH",
					 "VB", "VBD", "VBG", "VBN", "VBP",
					 "VBZ", "WDT", "WP", "WP$", "WRB"]

	for e in entries:
		text_id_list.append(e[0])
		post_test_list.append(e[1])
		click_bait_list.append(e[2])
		tweet_length_count_list.append(e[3])

	tweet_length_count_list = np.array(tweet_length_count_list)

	i = 0
	for text_tokens in post_test_list:
		pos_list = []
		# token = word_tokenize(text_tokens)
		token = regexp_tokenize(text_tokens, r"\w+")
		tagged = nltk.pos_tag(token)
		# print(tagged)
		for e in tagged:
			pos_list.append(e[1])
		# print(pos_list)

		for j, item in zip(range(0, 35), pos_item_list):
			pos_count[i][j] += pos_list.count(item)
		if tweet_length_count_list[i] == 0:
			pos_count_percentage[i] = 0
		else:
			pos_count_percentage[i] = pos_count[i] / tweet_length_count_list[i]
		i += 1

	sum_pos_count_click_bait = np.zeros(35, dtype=np.int)
	sum_pos_count_non_click_bait = np.zeros(35, dtype=np.int)

	for a, b in zip(click_bait_list, pos_count):
		if a == 1:
			sum_pos_count_click_bait += b
		elif a == 0:
			sum_pos_count_non_click_bait += b

	print(sum(sum_pos_count_click_bait), sum(sum_pos_count_non_click_bait))

	pos_count_6_cat = np.zeros((19463, 6), dtype=np.double)

	for i in range(0, 19463):
		if tweet_length_count_list[i] == 0:
			pos_count_6_cat[i][0] = pos_count_6_cat[i][1] = pos_count_6_cat[i][2] = \
				pos_count_6_cat[i][3] = pos_count_6_cat[i][4] = pos_count_6_cat[i][5] = 0
		else:
			pos_count_6_cat[i][0] = (pos_count[i][11] + pos_count[i][12] + pos_count[i][13] + pos_count[i][14] +
									 pos_count[i][16]) / int(tweet_length_count_list[i])
			pos_count_6_cat[i][1] = (pos_count[i][25] + pos_count[i][26] + pos_count[i][27] + pos_count[i][28] +
									 pos_count[i][29] + pos_count[i][30] + pos_count[i][10]) / int(
				tweet_length_count_list[i])
			pos_count_6_cat[i][2] = (pos_count[i][23] + pos_count[i][5] + pos_count[i][0] + pos_count[i][22]) / int(
				tweet_length_count_list[i])
			pos_count_6_cat[i][3] = (pos_count[i][15] + pos_count[i][2] + pos_count[i][17] + pos_count[i][18] +
									 pos_count[i][31] + pos_count[i][32] + pos_count[i][33] + pos_count[i][34] +
									 pos_count[i][3]) / int(tweet_length_count_list[i])
			pos_count_6_cat[i][4] = (pos_count[i][19] + pos_count[i][20] + pos_count[i][21] + pos_count[i][6] +
									 pos_count[i][7] + pos_count[i][8]) / int(tweet_length_count_list[i])
			pos_count_6_cat[i][5] = (pos_count[i][24] + pos_count[i][4] + pos_count[i][9] + pos_count[i][1]) / int(
				tweet_length_count_list[i])

	for i in range(0, 19463):
		cur.execute("update features set noun_pct = ?, verb_pct = ?, preposition_pct = ?, "
					"function_pct = ?, qualifier_pct = ?, others_pct = ?"
					" where text_id = ?", (pos_count_6_cat[i][0], pos_count_6_cat[i][1], pos_count_6_cat[i][2],
										   pos_count_6_cat[i][3], pos_count_6_cat[i][4], pos_count_6_cat[i][5],
										   text_id_list[i],))

	db_connection.commit()
	cur.close()
	db_connection.close()


def pos_analysis(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select noun_pct, verb_pct, preposition_pct, qualifier_pct, 
					function_pct, others_pct, click_bait from features;'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]
	entries = np.array(entries)

	df = pd.DataFrame(entries)
	df.columns = ["noun_pct", "verb_pct", "preposition_pct",
				  "qualifier_pct", "function_pct", "others_pct", "click_bait"]
	df["noun_pct"] = (df["noun_pct"] * 10).round()
	df["verb_pct"] = (df["verb_pct"] * 10).round()
	df["preposition_pct"] = (df["preposition_pct"] * 10).round()
	df["qualifier_pct"] = (df["qualifier_pct"] * 10).round()
	df["function_pct"] = (df["function_pct"] * 10).round()
	df["others_pct"] = (df["others_pct"] * 10).round()

	df_cb = df[(df["click_bait"] == 1)]
	df_ncb = df[(df["click_bait"] == 0)]
	df_ncb = df_ncb.sample(n=4700)

	df = pd.concat([df_cb, df_ncb], axis=0)

	df1 = df.groupby("noun_pct")["click_bait"].value_counts().unstack()
	df1 = df1.divide(df1.sum(axis=0), axis=1)
	pct_plot1 = df1.plot(kind="bar", stacked=False)
	pct_plot1.set_xlabel("Noun PoS Percentage: NPP")
	pct_plot1.set_ylabel("% of Headlines of NPP")
	pct_plot1.set_title(label="global 'non-cb' and 'cb' distribution")

	df1 = df1.divide(df1.sum(axis=1), axis=0)
	pct_plot2 = df1.plot(kind="bar", stacked=False)
	pct_plot2.set_xlabel("Noun PoS Percentage: NPP")
	pct_plot2.set_ylabel("Composition with NPP")
	pct_plot2.set_title(label="global 'non-cb' and 'cb' distribution")

	df2 = df.groupby("verb_pct")["click_bait"].value_counts().unstack()
	df2 = df2.divide(df2.sum(axis=0), axis=1)
	pct_plot3 = df2.plot(kind="bar", stacked=False)
	pct_plot3.set_xlabel("Verb PoS Percentage: VPP")
	pct_plot3.set_ylabel("% of Headlines of VPP")
	pct_plot3.set_title(label="global 'non-cb' and 'cb' distribution")

	df2 = df2.divide(df2.sum(axis=1), axis=0)
	pct_plot4 = df2.plot(kind="bar", stacked=False)
	pct_plot4.set_xlabel("Verb PoS Percentage: VPP")
	pct_plot4.set_ylabel("Composition with VPP")
	pct_plot4.set_title(label="global 'non-cb' and 'cb' distribution")

	df3 = df.groupby("preposition_pct")["click_bait"].value_counts().unstack()
	df3 = df3.divide(df3.sum(axis=0), axis=1)
	pct_plot5 = df3.plot(kind="bar", stacked=False)
	pct_plot5.set_xlabel("Preposition PoS Percentage: PPP")
	pct_plot5.set_ylabel("% of Headlines of PPP")
	pct_plot5.set_title(label="global 'non-cb' and 'cb' distribution")

	df3 = df3.divide(df3.sum(axis=1), axis=0)
	pct_plot6 = df3.plot(kind="bar", stacked=False)
	pct_plot6.set_xlabel("Preposition PoS Percentage: PPP")
	pct_plot6.set_ylabel("Composition with PPP")
	pct_plot6.set_title(label="global 'non-cb' and 'cb' distribution")

	df4 = df.groupby("qualifier_pct")["click_bait"].value_counts().unstack()
	df4 = df4.divide(df4.sum(axis=0), axis=1)
	pct_plot7 = df4.plot(kind="bar", stacked=False)
	pct_plot7.set_xlabel("Qualifier PoS Percentage: QPP")
	pct_plot7.set_ylabel("% of Headlines of QPP")
	pct_plot7.set_title(label="global 'non-cb' and 'cb' distribution")

	df4 = df4.divide(df4.sum(axis=1), axis=0)
	pct_plot8 = df4.plot(kind="bar", stacked=False)
	pct_plot8.set_xlabel("Qualifier PoS Percentage: QPP")
	pct_plot8.set_ylabel("Composition with QPP")
	pct_plot8.set_title(label="global 'non-cb' and 'cb' distribution")

	df5 = df.groupby("function_pct")["click_bait"].value_counts().unstack()
	df5 = df5.divide(df5.sum(axis=0), axis=1)
	pct_plot9 = df5.plot(kind="bar", stacked=False)
	pct_plot9.set_xlabel("Function PoS Percentage: FPP")
	pct_plot9.set_ylabel("% of Headlines of FPP")
	pct_plot9.set_title(label="global 'non-cb' and 'cb' distribution")

	df5 = df5.divide(df5.sum(axis=1), axis=0)
	pct_plot10 = df5.plot(kind="bar", stacked=False)
	pct_plot10.set_xlabel("Function PoS Percentage: FPP")
	pct_plot10.set_ylabel("Composition with FPP")
	pct_plot10.set_title(label="global 'non-cb' and 'cb' distribution")

	df6 = df.groupby("others_pct")["click_bait"].value_counts().unstack()
	df6 = df6.divide(df6.sum(axis=0), axis=1)
	pct_plot11 = df6.plot(kind="bar", stacked=False)
	pct_plot11.set_xlabel("Other PoS Percentage: OPP")
	pct_plot11.set_ylabel("% of Headlines of OPP")
	pct_plot11.set_title(label="global 'non-cb' and 'cb' distribution")

	df6 = df6.divide(df6.sum(axis=1), axis=0)
	pct_plot12 = df6.plot(kind="bar", stacked=False)
	pct_plot12.set_xlabel("Other PoS Percentage: OPP")
	pct_plot12.set_ylabel("Composition with OPP")
	pct_plot12.set_title(label="global 'non-cb' and 'cb' distribution")

	plt.show()

	print("deviation of 'noun_pct' of clickbait tweet:")
	print(df_cb["noun_pct"].std(axis=0))
	print("deviation of 'noun_pct' of non-clickbait tweet:")
	print(df_ncb["noun_pct"].std(axis=0))

	print("deviation of 'verb_pct' of clickbait tweet:")
	print(df_cb["verb_pct"].std(axis=0))
	print("deviation of 'verb_pct' of non-clickbait tweet:")
	print(df_ncb["verb_pct"].std(axis=0))

	print("deviation of 'preposition_pct' of clickbait tweet:")
	print(df_cb["preposition_pct"].std(axis=0))
	print("deviation of 'preposition_pct' of non-clickbait tweet:")
	print(df_ncb["preposition_pct"].std(axis=0))

	print("deviation of 'qualifier_pct' of clickbait tweet:")
	print(df_cb["qualifier_pct"].std(axis=0))
	print("deviation of 'qualifier_pct' of non-clickbait tweet:")
	print(df_ncb["qualifier_pct"].std(axis=0))

	print("deviation of 'function_pct' of clickbait tweet:")
	print(df_cb["function_pct"].std(axis=0))
	print("deviation of 'function_pct' of non-clickbait tweet:")
	print(df_ncb["function_pct"].std(axis=0))

	print("deviation of 'others_pct' of clickbait tweet:")
	print(df_cb["others_pct"].std(axis=0))
	print("deviation of 'others_pct' of non-clickbait tweet:")
	print(df_ncb["others_pct"].std(axis=0))


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()


pos_analysis(file_directory)
# pos(file_directory)
