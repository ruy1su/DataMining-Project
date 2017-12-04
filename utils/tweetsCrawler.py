# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
from utils import tools

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main(argv):
    keywords={"Apple","Google","Microsoft"}
    folders={"AAPL","GOOG","MSFT"}
    until=tools.nextDay(argv[1])
    since=argv[0]
    while since!= until:
    
        for word,folder in zip(keywords,folders):
	
		#opts, args = getopt.getopt(argv, "", ("username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))
                tweetCriteria = got.manager.TweetCriteria()
                nextDay=tools.nextDay(since)
		outputFileName = folder+"-"+since+".csv"
		
		tweetCriteria.since = since
		tweetCriteria.until =nextDay
		tweetCriteria.querySearch =word	
		tweetCriteria.maxTweets = 1000
		

		dirr="../data-set/tweets/"+folder+"/"+outputFileName
				
		outputFile = codecs.open(dirr, "w+", "utf-8")

		outputFile.write('retweets;favorites;text;mentions;hashtags')

		print('Searching...\n')
                
		def receiveBuffer(tweets):
                        
			for t in tweets:
				outputFile.write(('\n%d;%d;%s;%s;%s' % ( t.retweets,t.favorites,t.text,t.mentions, t.hashtags)))
				#print (t.username, t.date.strftime("%Y-%m-%d %H:%M"),t.text)
				
			outputFile.flush();

			print('More %d saved on file...\n' % len(tweets))
		got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)
		outputFile.close()
		print('Done. Output file generated "%s".' % outputFileName)
		
		
        since=nextDay
	
  
if __name__ == '__main__':
	main(sys.argv[1:])
