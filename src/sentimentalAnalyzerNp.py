from textblob import TextBlob
import csv
import sys
import os
import re
from datetime import datetime

def main():
    folders = ['AAPL', 'GOOG', 'MSFT']
    for folder in folders:
        writeFileName = "../data-set/sentiments/" + folder + "-mood-pn.csv"
        datafolder = "../data-set/tweets/" + folder +"/"
        with open(writeFileName, 'w') as outfile:
            outfile.write('date,polarity\n')
            for datafile in os.listdir(datafolder):
                match =  re.search(r'\d{4}-\d{2}-\d{2}', datafile)
                date = datetime.strptime(match.group(), '%Y-%m-%d').date()
            
                with open("../data-set/tweets/" + folder + "/" + datafile) as csvfile:
                    reader = csv.DictReader(csvfile,delimiter=';')
                    count = 0
                    sentiment = 0.0
                    for row in reader:
                        count += 1
                        text = row['text'].decode('ascii', errors="replace")
                        #print text
                        textblob = TextBlob(text)
                        sentiment += textblob.sentiment.polarity
                        sentiment = sentiment / count
                csvfile.close()
                outfile.write(str(date) + ',' + str(sentiment) + '\n')
        outfile.close
        '''
            for x in range (1, 32):
                if x < 10:
                    readFileName = "../data-set/"+ argv + "/" + argv+ "-2010-01-0" + str(x) + ".csv"
                else:
                    readFileName = "../data-set/"+ argv + "/" + argv+ "-2010-01-" + str(x) + ".csv"
                sentiment = 0.0
                with open(readFileName) as csvfile:
                    reader = csv.DictReader(csvfile,delimiter=';')
                    count = 1
                    for row in reader:
                        count += 1
                        text = row['text'].decode('ascii', errors="replace")
                        #print text
                        textblob = TextBlob(text)
                        sentiment += textblob.sentiment.polarity
                        #print textblob.sentiment.polarity
                    sentiment = sentiment / count
                    if x < 10:
                        date = "2010-01-0" + str(x)
                    else:
                        date = "2010-01-" + str(x)
            
            
                csvfile.close()
                outfile.write(date)
                outfile.write('\t')
                outfile.write(str(sentiment))
                outfile.write('\n')
                csvfile.close()
            outfile.close()
            '''

if __name__ == '__main__':
    main( )
