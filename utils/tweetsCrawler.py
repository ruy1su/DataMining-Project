# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs


if sys.version_info[0] < 3:
    import got
else:
    import got3 as got



   

def main(argv):
    def nextDay(date="", year=0, month=0, day=0):
        if(date != ""):
            date = date.split('-')
            year = eval(date[0])
            month = eval(date[1] if date[1][0] != '0' else date[1][1])
            day = eval(date[2] if date[2][0] != '0' else date[2][1])

            month_days = [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                          [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
            # if is_leap == 0, indicates current year is not leap year
            is_leap = 0 if (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)) else 1

            # update day, month, year
            day = 1 if (month_days[is_leap][month - 1] == day) else day + 1
            month = 1 if (day == 1 and month == 12) else (month + 1 if (day == 1) else month)
            year = year + 1 if (month == 1 and day == 1) else year
    
        return ("%04d-%02d-%02d" % (year, month, day))
    keywords={"Apple","Google","Microsoft"}
    folders={"AAPL","GOOG","MSFT"}
    until=nextDay(argv[1])
    since=argv[0]
    while since!= until:
    
        for word,folder in zip(keywords,folders):
	
		#opts, args = getopt.getopt(argv, "", ("username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))
                tweetCriteria = got.manager.TweetCriteria()
                nextDay=nextDay(since)
		outputFileName = folder + "-" + since + ".csv"
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
