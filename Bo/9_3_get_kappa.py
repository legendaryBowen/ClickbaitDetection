import sqlite3
import os
from fleiss_kappa import fleissKappa


def get_kappa(file_directory):
	#  database
	db_path = file_directory + "\\click_baitX.sqlite"
	db_connection = sqlite3.connect(db_path)
	cur = db_connection.cursor()

	sql = '''select count_0, count_033, count_066, count_1 from Fleiss_Kappa'''
	rate = cur.execute(sql)
	rate = [list(e[:]) for e in rate]


	kappa = fleissKappa(rate, 5)
	return kappa


os.chdir(os.path.dirname(__file__))
file_directory = os.getcwd()

kappa_value = get_kappa(file_directory)
print(kappa_value)