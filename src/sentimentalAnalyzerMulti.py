import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from csv import writer
import glob
import sys
import os
import re
from datetime import datetime

def read():
	readFromPath('AAPL/')
	readFromPath('GOOG/')
	readFromPath('MSFT/')

def readFromPath(dirStr):
	dirStrWoSlash = dirStr[:-1]
	defaultDir = '../data-set/tweets/'
	with open('../data-set/sentiments/'+dirStrWoSlash+'-mood-multi.csv', 'w') as fout:
		#data names
		fout.write('date,alert,happy,calm,compound\n')
		for fle in os.listdir(defaultDir + dirStr):
			#Getting date
			match =  re.search(r'\d{4}-\d{2}-\d{2}', fle)
			date = datetime.strptime(match.group(), '%Y-%m-%d').date()
			sentDim = []
			openPath = defaultDir + dirStr + fle
			# print openPath
			sentDim = getTwt(openPath)
			fout.write('%s,%s,%s,%s,%s\n' % (str(date), sentDim[0], sentDim[1], sentDim[2], sentDim[3]))
	fout.close()		
def getTwt(openPath):

    with open(openPath) as fin:
        twts = []
        firstline = True
        for line in fin:
        	#skip 1st line
        	if firstline:    
        		firstline = False
        	else:
				ls = line.rstrip().split(';')
				twt = ls[2]
				twts.append(twt)
            	# fout.write('%s\t%s\t%s\n' % (id0, id1, '1'))
            	# fout.write('%s\t%s\t%s\n' % (id1, id0, '1'))
    print twts
    return sentimentAnalyzer(twts)

def sentimentAnalyzer(twts):
    test = ["Great place to be when you are in Bangalore.",
             "The place was being renovated when I visited so the seating was limited.",
             "Loved the ambience, loved the food",
             "The food is delicious but not over the top.",
             "Service - Little slow, probably because too many people.",
             "The place is not easy to locate",
             "Mushroom fried rice was tasty",
             "hartzprod Slightly weird. I hope nothing comes up if you google me nude. lol",
             "My rank is very high, that's awesome"]
    
    sentDim = {'neg':'Alert','pos':'Happy', 'neu':'Calm','compound':'compound'}
    
    sentiment4OneDay4OneCom = []
    sentimentAlert = []
    sentimentCalm = []
    sentimentHappy = []
    sentimentCompound = []

    sid = SentimentIntensityAnalyzer()
    # assert(len(twts)>0)
    if len(twts) == 0:
        return [0.5,0.5,0.5,0.5]
    for sentence in twts:
        print(sentence+'\n')
        i = 0
        ss = sid.polarity_scores(sentence)
        for k in ss:
            print('{0}: {1}, '.format(sentDim[k], ss[k]))
            if sentDim[k] == 'Alert':
            	sentimentAlert.append(ss[k])
            if sentDim[k] == 'Calm':
            	sentimentCalm.append(ss[k])
            if sentDim[k] == 'compound':
            	sentimentCompound.append(ss[k])
            if sentDim[k] == 'Happy':
           		sentimentHappy.append(ss[k])
    # print (sentimentHappy)
    sentiment4OneDay4OneCom.append(sum(sentimentAlert) / float(len(sentimentAlert)))	
    sentiment4OneDay4OneCom.append(sum(sentimentHappy) / float(len(sentimentHappy)))
    sentiment4OneDay4OneCom.append(sum(sentimentCalm) / float(len(sentimentCalm)))
    sentiment4OneDay4OneCom.append(sum(sentimentCompound) / float(len(sentimentCompound)))
    # print (sentiment4OneDay4OneCom)
    return sentiment4OneDay4OneCom

# getTwt()
# sentimentAnalyzer()
read()
