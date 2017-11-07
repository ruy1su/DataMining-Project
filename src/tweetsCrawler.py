from tweepy.streaming import StreamListener
import tweepy
import json
import time
import io
import pprint as pp

consumer_key="PVUSRJzwwdoLiTaXnurcuF2FY"
consumer_secret="uGvBfc0f1nLA8e1LI5G5fqIak7GRjsYDLOfA09jePHy8YZ2Xr0"
access_token="924383751198478337-WPaL7r1fSe7WSsewzECVf1aAfwHHREE"
access_token_secret="aQlkuSWdDh7Iq6GpqV4Wzn8uDFbMAxL4lJFbNMwZy2SrZ"

class MyStreamListener(StreamListener):
    def __init__(self, time_limit=30):
        self.start = time.time()
        self.time_limit = time_limit
        self.outfile = open('sample_tweets_streaming.json', 'w')
        super(MyStreamListener, self).__init__()
    
    def on_data(self, data):
        if (time.time() - self.start < self.time_limit):
            parsed_data = json.loads(data)
            #str_to_print = (parsed_data["text"]).encode('utf8', 'replace')
            #self.outfile.write(str_to_print)
            json.dump(parsed_data["text"], self.outfile)
            self.outfile.write("\n")
            return True
        else:
            self.outfile.close()
            return False

def on_error(self, status):
    print(status)
    
    def on_status(self, status):
        print(status.text)

if __name__ == '__main__':
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    '''
    #---------------------------Streaming API---------------------------#
    myStreamListener = MyStreamListener(5)
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(track=["stocks"], languages=['en'])
    '''
    
    #---------------------------Search API---------------------------#
    results = api.search(q="stocks", count=100)

    with open('../sample_tweets.json', 'r+') as f:
        # read previous data
        prev_data = json.load(f)
        print "previous sample number: ", len(prev_data)

        # get current data
        tweetlist = []
        for r in results:

            #-----------------JSON format output -----------------#
            tweet_json = {
                'time': r.created_at.strftime('%Y-%m-%dT%H:%M:%S'),
                'tweet': r.text
            }
            tweetlist.append(tweet_json)
            
            '''
            #-----------------Line format output -----------------#
            json.dump(r.created_at.strftime('%Y-%m-%dT%H:%M:%S'), f)
            f.write('\t')
            json.dump(tweet_json, f)
            f.write('\n')
            '''
        new_data = prev_data + tweetlist
        print "current sample number: ", len(new_data)

        f.seek(0)
        json.dump(new_data, f, indent=4)
    f.close()
    
