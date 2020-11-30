import sqlite3
import os


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()


def import_value(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	cur.executescript('''
		Drop table if EXISTS Fleiss_Kappa;
		create TABLE Fleiss_Kappa(
			id integer not null unique primary key autoincrement,
			text_id text not null unique,
			count_0 integer,
			count_033 INTEGER,
			count_066 INTEGER,
			count_1 INTEGER);
	''')

	sql = '''select text_id, annotator1, annotator2, annotator3, annotator4, annotator5 from main'''
	entries = cur.execute(sql)
	entries = [list(e[:]) for e in entries]

	count_0 = 0
	count_033 = 0
	count_066 = 0
	count_1 = 0
	for e in entries:
		text_id = e[0]
		count_0 = e.count(0)
		count_033 = e.count(0.3333333333)
		count_066 = e.count(0.6666666666)
		count_1 = e.count(1)
		cur.execute("insert or ignore into Fleiss_Kappa(text_id, count_0, count_033, count_066, count_1) "
					"values(?, ?, ?, ?, ?)", (text_id, count_0, count_033, count_066, count_1))

	db_connection.commit()
	cur.close()
	db_connection.close()


import_value(file_directory)
