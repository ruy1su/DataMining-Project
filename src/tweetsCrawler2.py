# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main(argv):
    keywords={"Apple"} #,"Google","Microsoft"}
    folders={"AAPL"}#,"GOOG","MSFT"}
    month=str(argv[1])
    year=argv[0]
    day=01
    totalDays=30
    if month=='01' or month=='03' or month=='05' or month=='07'or month=='08'or month=='10' or month=='12': totalDays=31
    if month=='02': totalDays=29
    while day <= totalDays:
    
        for word,folder in zip(keywords,folders):
	
		#opts, args = getopt.getopt(argv, "", ("username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))
                tweetCriteria = got.manager.TweetCriteria()
                date1=str(year)+"-"+str(month)+"-"+str(day).zfill(2)
                if day==totalDays : date2=str(year)+"-"+str(int(month)+1)+"-"+str(1).zfill(2)
                else:  date2=str(year)+"-"+str(month)+"-"+str (day+1).zfill(2)
		outputFileName = folder+"-"+date1 +".csv"
		
		tweetCriteria.since = date1
		tweetCriteria.until =date2
		tweetCriteria.querySearch =word	
		tweetCriteria.maxTweets = 1000
		

		dirr="../data-set/tweets"+folder+"/"+outputFileName
				
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
		
		
        day+=1
	
  
if __name__ == '__main__':
	main(sys.argv[1:])
