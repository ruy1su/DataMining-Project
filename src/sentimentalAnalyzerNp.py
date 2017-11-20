from textblob import TextBlob
import csv
import sys


def main(argvs):
    for argv in argvs:
       
        writeFileName = "./data-set/sentiments/" + argv + "-mood-pn.csv"
        with open(writeFileName, 'w') as outfile:
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


if __name__ == '__main__':
    main(sys.argv[1:] )
