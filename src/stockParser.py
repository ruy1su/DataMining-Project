#!/usr/local/bin/python3

import csv

AAPL_PATH = "../data-set/AAPL.csv"
GOOG_PATH = "../data-set/GOOG.csv"
MSFT_PATH = "../data-set/MSFT.csv"

class stockParser(object):

	def __init__(self, file_path):
		self.stock_history_ = []

		with open(file_path, "r") as cvsfile:
			stock_reader = csv.reader(cvsfile)

			next(stock_reader)
			for row in stock_reader:
				#  0    1    2    3   4     5         6
				# data open high low close adj_close volumn
				self.stock_history_.append([row[0]] + [eval(x) for x in row[1:]])

		# Sort list in ascending order with date
		self.stock_history_ = sorted(self.stock_history_, key=lambda obj:obj[0])


	def getData(self):
		return self.stock_history_


	def size(self):
		return len(self.stock_history_)


	def calcFulctuation(self, i):
		if(i == 0):
			return None
		else:
			return (self.stock_history_[i - 1][4] - self.stock_history_[i][4]) / self.stock_history_[i - 1][4]

def main():
	stock_parser = stockParser(GOOG_PATH)
	for i in range(1, stock_parser.size()):
		print(stock_parser.calcFulctuation(i))

if __name__ == '__main__':
	main()