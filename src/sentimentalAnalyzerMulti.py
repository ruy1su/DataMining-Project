import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from csv import writer
import codecs
import glob
import sys
import os
import re
import indicoio
from datetime import datetime
indicoio.config.api_key = '67c2a07accd80ed593cdbbf089ed25ad'

class sentimentAnalyzerMultiDims:

    def read(self):
    	self.readFromPath('AAPL/')
    	self.readFromPath('GOOG/')
    	self.readFromPath('MSFT/')

    def readFromPath(self,dirStr):
    	dirStrWoSlash = dirStr[:-1]
    	defaultDir = '../data-set/tweets/'
    	with open('../data-set/sentiments/'+dirStrWoSlash+'-mood-multi-indico.csv', 'w') as fout:
    		#data names
    		# fout.write('date,alert,happy,calm,compound\n')
            fout.write('date,alert,happy,sad,surprise\n')
            for fle in os.listdir(defaultDir + dirStr):
    			#Getting date
                match =  re.search(r'\d{4}-\d{2}-\d{2}', fle)
                date = datetime.strptime(match.group(), '%Y-%m-%d').date()
                sentDim = []
                openPath = defaultDir + dirStr + fle
    			# print openPath
                sentDim = self.getTwt(openPath)
                print (sentDim,'sentDim')
                fout.write('%s,%s,%s,%s,%s\n' % (str(date), sentDim[0], sentDim[1], sentDim[2], sentDim[3]))
    	fout.close()		
    def getTwt(self,openPath):

        with codecs.open(openPath,"r",encoding='utf-8', errors='ignore') as fin:
            twts = []
            firstline = True
            for line in fin:
            	#skip 1st line
            	if firstline:    
            		firstline = False
            	else:
                    # ls = unicode(line, errors='replace')                     #Only read english sentences
                    # ls = line.decode('ascii', errors="replace")
                    ls = line.rstrip().split(';')
                    twt = ls[2]
                    twts.append(twt)
        # print (twts)
        return self.sentimentAnalyzerUsingIndicoio(twts[:5])

    def sentimentAnalyzerUsingWordList(self,twts):
        sentDim = {'Alert':0, 'Happy':0, 'Calm':0, 'Kind':0}
        alertList = ['bad', 'BITCH', 'careful', 'cautious', 'chary', 'circumspect', 'conservative', 'considerate', 'gingerly', 'guarded', 'heedful', 'safe', 'wary','advertent', 'attentive', 'awake', 'observant', 'regardful', 'vigilant', 'watchful', 'hypercautious', 'foresighted', 'foresightful', 'forethoughtful','provident','thoughtful','cagey','cagy','calculating','canny','shrewd','deliberate','slow','ultracareful','ultracautious','alive', 'aware', 'conscious', 'sensitive', 'cognizant', 'heedful', 'mindful', 'observing', 'regardful', 'sharp', 'sharp-eyed', 'hyperalert', 'hypervigilant', 'sleepless', 'wakeful', 'chary', 'prepared']
        calmList = ['abate','alleviate','assuage','calm','compose','decrease','lessen','lighten','make nice','mitigate','moderate','mollify','pacify','play up to','pour oil on','quiet','square','take the bite out','harmonious','low-key','mild','placid','serene','slow','smooth','soothing','tranquil','bucolic','halcyon','hushed','pacific','pastoral','reposing','still','at a standstill','at peace','bland','breathless','breezeless','in order','inactive','motionless','quiescent','reposeful','restful','rural','stormless','undisturbed','unruffled','waveless','windless']
        kindList = ['affectionate','amiable','charitable','compassionate','considerate','cordial','courteous','friendly','gentle','gracious','humane','kindhearted','kindly','loving','sympathetic','thoughtful','tolerant','humanitarian','understanding','all heart','altruistic','amicable','beneficent','benevolent','benign','big','bleeding-heart','bounteous','clement','congenial','eleemosynary','good-hearted','heart in right place','indulgent','lenient','mild','neighborly','obliging','philanthropic','propitious','soft touch','softhearted','tenderhearted']
        happyList = ['love','like', 'adore','affable','aggreeable','amiable','amusing','animated','appealing','happy','beaming','beatific','beautiful','bliss','blissful','blithe','bowl over','buoyant','carefree','charming','cheerful','cheeriness','cheery','chipper','chirpy','content','cool','contented','delight','delighted','delightful','diverting','droll','ebullient ','ecstasy','ecstatic','elated','elation','enchanting','endearing','energized','engaging','enjoyable','entertaining','euphoria','euphoric','excited','exhilirated','exuberance','exultant','exultation','favorable','fine','friendly','fulfilled','fun','funny','genial','glad','gladden','glee','glory','glory in','glorious','good','good humored','good mood','good natured','grateful','gratified','gratify','gratifying','great','grinning','happiness','heartening','heartwarming','heavenly','high','high spirits','hilarious','hopeful','in a good mood','in good spirits','in seventh heaven','invigorated','jocular','jollity','joie de vivre','joyfulness','joyous','jubilation','jumping for joy','lap up','lighthearted','likable','looking on the bright side','lovable','lovely','lucky','luxuriate in','merriment','merry','mirth','mirthful','never been better','nice','obliging','on cloud nine','on top of the world','open','opportune','optomistic','overjoyed','over the moon','paradise','perkiness','perky','pleasant','pleased','please greatly','pleasure','precious','radiant','rapture','rapturous','relaxed','relish','revel in','satisfied','savor','simpatico','smiling','smart','source of pleasure','sparkle','stimulated','sunniness','sunny','sweet','take pleasure in','tears of joy','thrill','thrilled','tickled pink','touching','treat','triumph','upbeat','uplifting','untroubled','vitalized','vivacity','flower','sunl','walking on air','welcoming','willing','wondrous','zest for life','Laughter','laughed','laughing','excellent','laughs','successful','win','rainbow','smile','smiled','rainbows','winning','celebration','enjoyed','healthy','music','celebrating','congratulations','weekend','celebrate','comedy','jokes','rich','victory','Christmas','friendship','holidays','loved','loves','loving','beach','hahaha','kissing','sunshine','delicious','outstanding','sweetest','vacation','butterflies','freedom','ight','sweetheart','sweetness','award','chocolate','hahahaha','peace','splendid','enjoying','kissed','attraction','celebrated','hero','hugs','positive','birthday','blessed','fantastic','winner','beauty','butterfly','entertainment','funniest','honesty','sky','smiles','succeed','wonderful','kisses','promotion','family','gift','romantic','cupcakes','festival','hahahahaha','honour','weekends','angel','b-day','bonus','brilliant','diamonds','super','amazing','angels','profit','finest','bday','champion','grandmother','kitten','miracle','mom','blessings','cutest','excitement','millionaire','prize','succeeded','successfully','winners','shines','awesome','genius','achievement','cheers','exciting','goodness','income','party','puppy','song','succeeding','tasty','victories','achieved','billion','easier','flowers','gifts','gold','families','handsome','lovers','affection','candy','earnings','interesting','peacefully','praise','relaxing','roses','Saturdays','faithful','heavens','cherish','comfort','congrats','extraordinary','moonlight','optimistic','romance','feast','attractive','grandma','internet','profits']
        sentenceSum = 0
        sentiment4OneDay4OneCom = []
        if len(twts) == 0:
            return [0.5,0.5,0.5,0.5]
        for sentence in twts:
            sentenceSum+=1
            sentenceList = sentence.rstrip().split(' ')
            print (sentenceList,"sentenceList\n")
            print(sentence+'\n')
            for word in sentence:
                if word in alertList:
                    sentDim['Alert']+=1
                elif word in happyList:
                    sentDim['Happy']+=1
                elif word in calmList:
                    sentDim['Calm']+=1
                elif word in kindList:
                    sentDim['Kind']+=1
                # else:
                #     sentDim['Calm']+=1
        sentiment4OneDay4OneCom.append(sentDim['Alert'] / float(len(twts)))    
        sentiment4OneDay4OneCom.append(sentDim['Happy'] / float(len(twts)))
        sentiment4OneDay4OneCom.append(sentDim['Calm'] / float(len(twts)))
        sentiment4OneDay4OneCom.append(sentDim['Kind'] / float(len(twts)))
        print (sentiment4OneDay4OneCom)
        return sentiment4OneDay4OneCom

    def sentimentAnalyzer(self,twts):
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
        assert(len(twts)>=0)
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

    def sentimentAnalyzerUsingIndicoio(self, twts):
        sentDim = {'anger':'Alert','joy':'Happy', 'sadness':'Sad','fear':'Fear', 'surprise':'Surprise'}
        
        sentiment4OneDay4OneCom = []
        sentimentAlert = []
        sentimentSad = []
        sentimentHappy = []
        sentimentSurprise = []

        sid = SentimentIntensityAnalyzer()
        assert(len(twts)>=0)
        if len(twts) == 0:
            return [0.5,0.5,0.5,0.5]
        sums = 0
        for sentence in twts:
            print(sentence+'\n')
            i = 0
            sums+=1
            ss = indicoio.emotion(sentence)
            print (ss,sums, 'out of 1000')
            for k in ss:
                print (k)
                print('{0}: {1}, '.format(sentDim[k], ss[k]))
                if sentDim[k] == 'Alert':
                    sentimentAlert.append(ss[k])
                if sentDim[k] == 'Happy':
                    sentimentHappy.append(ss[k])
                if sentDim[k] == 'Sad':
                    sentimentSad.append(ss[k])
                if sentDim[k] == 'Surprise':
                    sentimentSurprise.append(ss[k])
        # print (sentimentHappy)
        sentiment4OneDay4OneCom.append(sum(sentimentAlert) / float(len(sentimentAlert)))    
        sentiment4OneDay4OneCom.append(sum(sentimentHappy) / float(len(sentimentHappy)))
        sentiment4OneDay4OneCom.append(sum(sentimentSurprise) / float(len(sentimentSurprise)))
        sentiment4OneDay4OneCom.append(sum(sentimentSad) / float(len(sentimentSad)))
        print (sentiment4OneDay4OneCom,'dsadsa')
        return sentiment4OneDay4OneCom


# read()
sent = sentimentAnalyzerMultiDims()
sent.read()