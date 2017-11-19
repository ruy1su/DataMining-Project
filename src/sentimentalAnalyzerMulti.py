import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
 

def getTwt():

    with open('../data-set/tweets/AAPL/AAPL-2010-01-01.csv') as fin, open('GOOG-mood-multi.csv', 'w') as fout:
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
            	# line = line.replace(",", "\t")
            	# fout.write('%s\t%s\t%s\n' % (id0, id1, '1'))
            	# fout.write('%s\t%s\t%s\n' % (id1, id0, '1'))
    print twts
    sentimentAnalyzer(twts)

def sentimentAnalyzer(twts):
    test = ["Great place to be when you are in Bangalore.",
             "The place was being renovated when I visited so the seating was limited.",
             "Loved the ambience, loved the food",
             "The food is delicious but not over the top.",
             "Service - Little slow, probably because too many people.",
             "The place is not easy to locate",
             "Mushroom fried rice was tasty",
             "hartzprod Slightly weird. I hope nothing comes up if you google me nude. lol"]
    sentDim = {'neg':'Alert','pos':'Happy', 'neu':'Calm','compound':'compound'}

    sid = SentimentIntensityAnalyzer()
    for sentence in twts:
        print(sentence+'\n')
        ss = sid.polarity_scores(sentence)
        for k in ss:
            print('{0}: {1}, '.format(sentDim[k], ss[k]))
        print()

getTwt()
# sentimentAnalyzer()
