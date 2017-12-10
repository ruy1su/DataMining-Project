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
import html
import HTMLParser
import random
from datetime import datetime
indicoio.config.api_key = 'c23f400f2986ffc2429d8b8b0529146a'

class sentimentAnalyzerMultiDims:
    mode = 0 #default
    modes = ['','-countWords','-indicoio']
    days = 100000000 #default dates we read from data
    todayDate = '2010/01/01' #default

    def read(self):
        # self.readFromPath('AAPL/')
        self.readFromPath('GOOG/')
        # self.readFromPath('MSFT/')

    def readFromPath(self,dirStr):
        print ('Company',dirStr)
        dirStrWoSlash = dirStr[:-1]
        defaultDir = '../data-set/tweets/'
        if self.days < 100:
            for fle in os.listdir(defaultDir + dirStr):
                #Getting date
                match =  re.search(r'\d{4}-\d{2}-\d{2}', fle)
                date = datetime.strptime(match.group(), '%Y-%m-%d').date()
                self.todayDate = str(date)
                sentDim = []
                openPath = defaultDir + dirStr + fle
                sentDim = self.getTwt(openPath)
        else:
            with open('../data-set/sentiments/'+dirStrWoSlash+'-mood-multi'+self.modes[int(mode)]+'.csv', 'w') as fout:
                if mode == 0: # Using Vader Lib
                    fout.write('date,alert,happy,calm,compound\n')
                elif mode == 1:# Using Word List
                    fout.write('date,alert,happy,calm,kind\n')
                elif mode == 2:#Using Indicoio
                    fout.write('date,alert,happy,sad,surprise\n')
                for fle in os.listdir(defaultDir + dirStr):
                    #Getting date
                    match =  re.search(r'\d{4}-\d{2}-\d{2}', fle)
                    date = datetime.strptime(match.group(), '%Y-%m-%d').date()
                    self.todayDate = str(date)
                    sentDim = []
                    openPath = defaultDir + dirStr + fle
                    sentDim = self.getTwt(openPath)
                    # print (sentDim,'sentDim')
                    fout.write('%s,%s,%s,%s,%s\n' % (str(date), sentDim[0], sentDim[1], sentDim[2], sentDim[3]))
            fout.close()    

    def getTwt(self,openPath):

        with codecs.open(openPath,"r",encoding='utf-8', errors='ignore') as fin:
            twts = []
            firstline = True
            looptime = 0
            for line in fin:
                #skip 1st line
                if looptime > self.days:
                    break
                looptime += 1 
                if firstline:    
                    firstline = False
                else:
                    # ls = unicode(line, errors='replace')                     #Only read english sentences
                    # ls = line.decode('ascii', errors="replace")
                    ls = line.rstrip().split(';')
                    if len(ls)<3:
                        twts.append('')
                    else:
                        twt = ls[2]
                        twts.append(twt)
            # print (len(twts),'length')
        if self.mode == 0:
            return self.sentimentAnalyzer(twts)
        elif self.mode == 1:
            return self.sentimentAnalyzerUsingWordList(twts)
        elif self.mode == 2:
            return self.sentimentAnalyzerUsingIndicoio(random.sample(twts, 10))
        else:
            print ("\n--------Wrong Arguements: sentimental mode should be 0,1 or 2-------\n")

    def sentimentAnalyzerUsingWordList(self,twts):      
        sentDim = {'Alert':0, 'Happy':0, 'Calm':0, 'Kind':0}
        alertList = ['abysmal','adverse','alarming','angry','annoy','anxious','apathy','appalling','atrocious','awful','bad','banal','barbed','belligerent','bemoan','beneath','boring','broken','callous',"can't",'clumsy','coarse','cold','cold-hearted','collapse','confused','contradictory','contrary','corrosive','corrupt','crazy','creepy','criminal','cruel','cry','cutting','dead','decaying','damage','damaging','dastardly','deplorable','depressed','deprived','deformed.','deny','despicable','detrimental','dirty','disease','disgusting','disheveled','dishonest','dishonorable','dismal','distress',"don't",'dreadful','dreary','enraged','eroding','evil','fail','faulty','fear','feeble','fight','filthy','foul','frighten','frightful','gawky','ghastly','grave','greed','grim','grimace','gross','grotesque','gruesome','guilty', 'wtf','haggard','hard','hard-hearted','harmful','hate','hideous','homely','horrendous','horrible','hostile','hurt','hurtful','icky','ignore','ignorant','ill','immature','imperfect','impossible','inane','inelegant','infernal','injure','injurious','insane','insidious','insipid','jealous','junky','lose','lousy','lumpy','malicious','mean','menacing','messy','misshapen','missing','misunderstood','moldy','monstrous','naive','nasty','naughty','negate','negative','never','nobody','nondescript','nonsense','not','noxious','objectionable','odious','offensive','oppressive','pain','perturb','pessimistic','petty','plain','poisonous','poor','prejudice','questionable','quirky','quit','reject','renege','repellant','reptilian','repulsive','repugnant','revenge','revolting','rocky','rotten','rude','ruthless','sad','savage','scare','scary','scream','severe','shoddy','shocking','sick','sickening','sinister','slimy','smelly','sobbing','sorry','spiteful','sticky','stinky','stormy','stressful','stuck','stupid','substandard','suspect','suspicious','tense','terrible','terrifying','threatening','ugly','undermine','unfair','unfavorable','unhappy','unhealthy','unjust','unlucky','unpleasant','upset','unsatisfactory','unsightly','untoward','unwanted','unwelcome','unwholesome','unwieldy','unwise','vice','vicious','vile','villainous','vindictive','wary','weary','wicked','woeful','worthless','wound','yell','yucky','zero']
        calmList = ['abate','alleviate','assuage','calm','compose','decrease','lessen','lighten','make nice','mitigate','moderate','mollify','ok','okay','pacify','play up to','pour oil on','quiet','square','take the bite out','harmonious','low-key','mild','placid','serene','slow','smooth','soothing','tranquil','bucolic','halcyon','hushed','pacific','pastoral','reposing','still','at a standstill','at peace','bland','breathless','breezeless','in order','inactive','motionless','quiescent','reposeful','restful','rural','stormless','undisturbed','unruffled','waveless','windless']
        kindList = ['affectionate','amiable','charitable','compassionate','considerate','cordial','courteous','friendly','gentle','gracious','humane','kindhearted','kindly','loving','sympathetic','thoughtful','tolerant','humanitarian','understanding','all heart','altruistic','amicable','beneficent','benevolent','benign','big','bleeding-heart','bounteous','clement','congenial','eleemosynary','good-hearted','heart in right place','indulgent','lenient','mild','neighborly','obliging','philanthropic','propitious','soft touch','softhearted','tenderhearted']
        happyList = ['happy','abound','abounds','abundance','abundant','accessable','accessible','acclaim','acclaimed','acclamation','accolade','accolades','accommodative','accomodative','accomplish','accomplished','accomplishment','accomplishments','accurate','accurately','achievable','achievement','achievements','achievible','acumen','adaptable','adaptive','adequate','adjustable','admirable','admirably','admiration','admire','admirer','admiring','admiringly','adorable','adore','adored','adorer','adoring','adoringly','adroit','adroitly','adulate','adulation','adulatory','advanced','advantage','advantageous','advantageously','advantages','adventuresome','adventurous','advocate','advocated','advocates','affability','affable','affably','affectation','affection','affectionate','affinity','affirm','affirmation','affirmative','affluence','affluent','afford','affordable','affordably','afordable','agile','agilely','agility','agreeable','agreeableness','agreeably','all-around','alluring','alluringly','altruistic','altruistically','amaze','amazed','amazement','amazes','amazing','amazingly','ambitious','ambitiously','ameliorate','amenable','amenity','amiability','amiabily','amiable','amicability','amicable','amicably','amity','ample','amply','amuse','amusing','amusingly','angel','angelic','apotheosis','appeal','appealing','applaud','appreciable','appreciate','appreciated','appreciates','appreciative','appreciatively','appropriate','approval','approve','ardent','ardently','ardor','articulate','aspiration','aspirations','aspire','assurance','assurances','assure','assuredly','assuring','astonish','astonished','astonishing','astonishingly','astonishment','astound','astounded','astounding','astoundingly','astutely','attentive','attraction','attractive','attractively','attune','audible','audibly','auspicious','authentic','authoritative','autonomous','available','aver','avid','avidly','award','awarded','awards','awe','awed','awesome','awesomely','awesomeness','awestruck','awsome','backbone','balanced','bargain','beauteous','beautiful','beautifullly','beautifully','beautify','beauty','beckon','beckoned','beckoning','beckons','believable','believeable','beloved','benefactor','beneficent','beneficial','beneficially','beneficiary','benefit','benefits','benevolence','benevolent','benifits','best','best-known','best-performing','best-selling','better','better-known','better-than-expected','beutifully','blameless','bless','blessing','bliss','blissful','blissfully','blithe','blockbuster','bloom','blossom','bolster','bonny','bonus','bonuses','boom','booming','boost','boundless','bountiful','brainiest','brainy','brand-new','brave','bravery','bravo','breakthrough','breakthroughs','breathlessness','breathtaking','breathtakingly','breeze','bright','brighten','brighter','brightest','brilliance','brilliances','brilliant','brilliantly','brisk','brotherly','bullish','buoyant','cajole','calm','calming','calmness','capability','capable','capably','captivate','captivating','carefree','cashback','cashbacks','catchy','celebrate','celebrated','celebration','celebratory','champ','champion','charisma','charismatic','charitable','charm','charming','charmingly','chaste','cheaper','cheapest','cheer','cheerful','cheery','cherish','cherished','cherub','chic','chivalrous','chivalry','civility','civilize','clarity','classic','classy','clean','cleaner','cleanest','cleanliness','cleanly','clear','clear-cut','cleared','clearer','clearly','clears','clever','cleverly','cohere','coherence','coherent','cohesive','colorful','comely','comfort','comfortable','comfortably','comforting','comfy','commend','commendable','commendably','commitment','commodious','compact','compactly','compassion','compassionate','compatible','competitive','complement','complementary','complemented','complements','compliant','compliment','complimentary','comprehensive','conciliate','conciliatory','concise','confidence','confident','congenial','congratulate','congratulation','congratulations','congratulatory','conscientious','considerate','consistent','consistently','constructive','consummate','contentment','continuity','contrasty','contribution','convenience','convenient','conveniently','convience','convienient','convient','convincing','convincingly','cool','coolest','cooperative','cooperatively','cornerstone','correct','correctly','cost-effective','cost-saving','counter-attack','counter-attacks','courage','courageous','courageously','courageousness','courteous','courtly','covenant','cozy','creative','credence','credible','crisp','crisper','cure','cure-all','cushy','cute','cuteness','danke','danken','daring','daringly','darling','dashing','dauntless','dawn','dazzle','dazzled','dazzling','dead-cheap','dead-on','decency','decent','decisive','decisiveness','dedicated','defeat','defeated','defeating','defeats','defender','deference','deft','deginified','delectable','delicacy','delicate','delicious','delight','delighted','delightful','delightfully','delightfulness','dependable','dependably','deservedly','deserving','desirable','desiring','desirous','destiny','detachable','devout','dexterous','dexterously','dextrous','dignified','dignify','dignity','diligence','diligent','diligently','diplomatic','dirt-cheap','distinction','distinctive','distinguished','diversified','divine','divinely','dominate','dominated','dominates','dote','dotingly','doubtless','dreamland','dumbfounded','dumbfounding','dummy-proof','durable','dynamic','eager','eagerly','eagerness','earnest','earnestly','earnestness','ease','eased','eases','easier','easiest','easiness','easing','easy','easy-to-use','easygoing','ebullience','ebullient','ebulliently','ecenomical','economical','ecstasies','ecstasy','ecstatic','ecstatically','edify','educated','effectively','effectiveness','effectual','efficacious','efficient','efficiently','effortless','effortlessly','effusion','effusive','effusively','effusiveness','elan','elate','elated','elatedly','elation','electrify','elegance','elegant','elegantly','elevate','elite','eloquence','eloquent','eloquently','embolden','eminence','eminent','empathize','empathy','empower','empowerment','enchant','enchanted','enchanting','enchantingly','encourage','encouragement','encouraging','encouragingly','endear','endearing','endorse','endorsed','endorsement','endorses','endorsing','energetic','energize','energy-efficient','energy-saving','engaging','engrossing','enhance','enhanced','enhancement','enhances','enjoy','enjoyable','enjoyably','enjoyed','enjoying','enjoyment','enjoys','enlighten','enlightenment','enliven','ennoble','enough','enrapt','enrapture','enraptured','enrich','enrichment','enterprising','entertain','entertaining','entertains','enthral','enthrall','enthralled','enthuse','enthusiasm','enthusiast','enthusiastic','enthusiastically','entice','enticed','enticing','enticingly','entranced','entrancing','entrust','enviable','enviably','envious','enviously','enviousness','envy','equitable','ergonomical','err-free','erudite','ethical','eulogize','euphoria','euphoric','euphorically','evaluative','evenly','eventful','everlasting','evocative','exalt','exaltation','exalted','exaltedly','exalting','exaltingly','examplar','examplary','excallent','exceed','exceeded','exceeding','exceedingly','exceeds','excel','exceled','excelent','excellant','excelled','excellence','excellency','excellent','excellently','excels','exceptional','exceptionally','excite','excited','excitedly','excitedness','excitement','excites','exciting','excitingly','exellent','exemplar','exemplary','exhilarate','exhilarating','exhilaratingly','exhilaration','exonerate','expansive','expeditiously','expertly','exquisite','exquisitely','extol','extoll','extraordinarily','extraordinary','exuberance','exuberant','exuberantly','exult','exultant','exultation','exultingly','eye-catch','eye-catching','eyecatch','eyecatching','fabulous','fabulously','facilitate','fair','fairly','fairness','faith','faithful','faithfully','faithfulness','fame','famed','famous','famously','fancier','fancinating','fancy','fanfare','fans','fantastic','fantastically','fascinate','fascinating','fascinatingly','fascination','fashionable','fashionably','fast','fast-growing','fast-paced','faster','fastest','fastest-growing','faultless','fav','fave','favor','favorable','favored','favorite','favorited','favour','fearless','fearlessly','feasible','feasibly','feature-rich','fecilitous','feisty','felicitate','felicitous','felicity','fertile','fervent','fervently','fervid','fervidly','fervor','festive','fidelity','fiery','fine','fine-looking','finely','finer','finest','firmer','first-class','first-in-class','first-rate','flashy','flatter','flattering','flatteringly','flawless','flawlessly','flexibility','flexible','flourish','flourishing','flutter','fond','fondly','fondness','foolproof','foremost','foresight','formidable','fortitude','fortuitous','fortuitously','fortunate','fortunately','fortune','fragrant','freed','freedom','freedoms','fresh','fresher','freshest','friendliness','friendly','frolic','frugal','fruitful','ftw','fulfillment','fun','futurestic','futuristic','gaiety','gaily','gained','gainful','gainfully','gaining','gains','gallant','gallantly','galore','geekier','geeky','gems','generosity','generous','generously','genius','gentle','gentlest','genuine','gifted','glad','gladden','gladly','gladness','glamorous','glee','gleeful','gleefully','glimmer','glimmering','glisten','glistening','glitter','glitz','glorify','glorious','gloriously','glory','glow','glowing','glowingly','god-given','god-send','godlike','godsend','gold','golden','good','goodly','goodness','goodwill','goood','gooood','gorgeous','gorgeously','grace','graceful','gracefully','gracious','graciously','graciousness','grand','grandeur','grateful','gratefully','gratification','gratified','gratifies','gratify','gratifying','gratifyingly','gratitude','great','greatest','greatness','grin','groundbreaking','guarantee','guidance','guiltless','gumption','gush','gusto','gutsy','hail','halcyon','hale','hallmark','hallmarks','hallowed','handier','handily','hands-down','handsome','handsomely','handy','happier','happily','happiness','happy','hard-working','hardier','hardy','harmless','harmonious','harmoniously','harmonize','harmony','headway','heal','healthful','healthy','hearten','heartening','heartfelt','heartily','heartwarming','heaven','heavenly','helped','helpful','helping','hero','heroic','heroically','heroine','heroize','heros','high-quality','high-spirited','hilarious','holy','homage','honest','honesty','honor','honorable','honored','honoring','hooray','hopeful','hospitable','hot','hotcake','hotcakes','hottest','hug','humane','humble','humility','humor','humorous','humorously','humour','humourous','ideal','idealize','ideally','idol','idolize','idolized','idyllic','illuminate','illuminati','illuminating','illumine','illustrious','ilu','imaculate','imaginative','immaculate','immaculately','immense','impartial','impartiality','impartially','impassioned','impeccable','impeccably','important','impress','impressed','impresses','impressive','impressively','impressiveness','improve','improved','improvement','improvements','improves','improving','incredible','incredibly','indebted','individualized','indulgence','indulgent','industrious','inestimable','inestimably','inexpensive','infallibility','infallible','infallibly','influential','ingenious','ingeniously','ingenuity','ingenuous','ingenuously','innocuous','innovation','innovative','inpressed','insightful','insightfully','inspiration','inspirational','inspire','inspiring','instantly','instructive','instrumental','integral','integrated','intelligence','intelligent','intelligible','interesting','interests','intimacy','intimate','intricate','intrigue','intriguing','intriguingly','intuitive','invaluable','invaluablely','inventive','invigorate','invigorating','invincibility','invincible','inviolable','inviolate','invulnerable','irreplaceable','irreproachable','irresistible','irresistibly','issue-free','jaw-droping','jaw-dropping','jollify','jolly','jovial','joyful','joyfully','joyous','joyously','jubilant','jubilantly','jubilate','jubilation','jubiliant','judicious','justly','keen','keenly','keenness','kid-friendly','kindliness','kindly','kindness','knowledgeable','kudos','large-capacity','laudable','laudably','lavish','lavishly','law-abiding','lawful','lawfully','lead','leading','leads','legendary','leverage','levity','liberate','liberation','liberty','lifesaver','light-hearted','lighter','likable','liked','likes','liking','lionhearted','lively','logical','long-lasting','lovable','lovably','loveliness','lovely','lover','loves','loving','low-cost','low-price','low-priced','low-risk','lower-priced','loyal','loyalty','lucid','lucidly','luck','luckier','luckiest','luckiness','lucky','lucrative','luminous','lush','luster','lustrous','luxuriant','luxuriate','luxurious','luxuriously','luxury','lyrical','magic','magical','magnanimous','magnanimously','magnificence','magnificent','magnificently','majestic','majesty','manageable','maneuverable','marvel','marveled','marvelled','marvellous','marvelous','marvelously','marvelousness','marvels','master','masterful','masterfully','masterpiece','masterpieces','masters','mastery','matchless','mature','maturely','maturity','meaningful','memorable','merciful','mercifully','mercy','merit','meritorious','merrily','merriment','merriness','merry','mesmerize','mesmerized','mesmerizes','mesmerizing','mesmerizingly','meticulous','meticulously','mightily','mighty','mind-blowing','miracle','miracles','miraculous','miraculously','miraculousness','modern','modest','modesty','momentous','monumental','monumentally','morality','motivated','multi-purpose','navigable','neat','neatest','neatly','nice','nicely','nicer','nicest','nifty','nimble','nobly','noiseless','non-violence','non-violent','notably','noteworthy','nourish','nourishing','nourishment','novelty','nurturing','oasis','obsession','obsessions','obtainable','openly','openness','optimal','optimism','optimistic','opulent','orderly','originality','outdo','outdone','outperform','outperformed','outperforming','outperforms','outshine','outshone','outsmart','outstanding','outstandingly','outstrip','outwit','overjoyed','overtake','overtaken','overtakes','overtaking','overtook','overture','pain-free','painless','painlessly','palatial','pamper','pampered','pamperedly','pamperedness','pampers','panoramic','paradise','paramount','pardon','passionately','patience','patient','patiently','patriot','patriotic','peace','peaceable','peaceful','peacefully','peacekeepers','peach','peerless','pep','pepped','pepping','peppy','peps','perfect','perfection','perfectly','permissible','perseverance','persevere','personages','personalized','phenomenal','phenomenally','picturesque','piety','pinnacle','playful','playfully','pleasant','pleasantly','pleased','pleases','pleasing','pleasingly','pleasurable','pleasurably','pleasure','plentiful','pluses','plush','plusses','poetic','poeticize','poignant','poise','poised','polished','polite','politeness','popular','portable','posh','positive','positively','positives','powerful','powerfully','praise','praiseworthy','praising','pre-eminent','precious','precise','precisely','preeminent','prefer','preferable','preferably','prefered','preferes','preferring','prefers','premier','prestige','prestigious','prettily','pretty','priceless','pride','principled','privilege','privileged','prize','proactive','problem-free','problem-solver','prodigious','prodigiously','prodigy','productive','productively','proficient','proficiently','profound','profoundly','profuse','profusion','progress','progressive','prolific','prominence','prominent','promise','promised','promises','promising','promoter','prompt','promptly','proper','properly','propitious','propitiously','pros','prosper','prosperity','prosperous','prospros','protect','protection','protective','proud','proven','providence','prowess','prudence','prudent','prudently','punctual','pure','purify','purposeful','quaint','qualified','qualify','quicker','quiet','quieter','radiance','radiant','rapid','rapport','raptureous','raptureously','rapturous','rapturously','razor-sharp','reachable','readable','readily','ready','reaffirm','reaffirmation','realistic','realizable','reasonable','reasonably','reasoned','reassurance','reassure','receptive','reclaim','recomend','recommend','recommendation','recommendations','recommended','reconcile','reconciliation','record-setting','recover','recovery','rectification','rectify','rectifying','redeem','redeeming','redemption','refine','refined','refinement','reform','reformed','reforming','reforms','refresh','refreshed','refreshing','refund','refunded','regal','regally','regard','rejoice','rejoicing','rejoicingly','rejuvenate','rejuvenated','rejuvenating','relaxed','relent','reliable','reliably','relief','relish','remarkable','remarkably','remedy','remission','remunerate','renaissance','renewed','renown','renowned','reputable','reputation','resilient','resolute','resound','resounding','resourceful','resourcefulness','respect','respectable','respectful','respectfully','respite','resplendent','responsibly','responsive','restful','restored','restructure','restructured','restructuring','retractable','revel','revelation','revere','reverence','reverent','reverently','revitalize','revival','revive','revives','revolutionary','revolutionize','revolutionized','revolutionizes','reward','rewarding','rewardingly','richer','richly','richness','righteous','righteously','righteousness','rightful','rightfully','rightly','rightness','risk-free','robust','rock-star','rock-stars','rockstar','rockstars','romantic','romantically','romanticize','roomier','roomy','rosy','safe','safely','sagacity','sagely','saint','saintliness','saintly','salutary','salute','sane','satisfactorily','satisfactory','satisfied','satisfies','satisfy','satisfying','satisified','savings','savior','savvy','scenic','seamless','seasoned','secure','securely','selective','self-determination','self-respect','self-satisfaction','self-sufficiency','self-sufficient','sensation','sensational','sensationally','sensations','sensible','sensibly','sensitive','serene','serenity','sexy','sharper','sharpest','shimmering','shimmeringly','shiny','significant','silent','simpler','simplest','simplified','simplifies','simplify','simplifying','sincere','sincerely','sincerity','skill','skilled','skillful','skillfully','slammin','sleek','slick','smarter','smartest','smartly','smile','smiles','smiling','smilingly','smitten','smooth','smoother','smoothes','smoothest','smoothly','snappy','snazzy','sociable','soft','softer','solace','solicitous','solicitously','solid','solidarity','soothe','soothingly','sophisticated','soulful','soundly','soundness','spacious','sparkle','sparkling','spectacular','spectacularly','speedily','speedy','spellbind','spellbinding','spellbindingly','spellbound','spiritual','splendid','splendidly','splendor','spontaneous','sporty','spotless','sprightly','stability','stabilize','stainless','standout','state-of-the-art','stately','statuesque','staunch','staunchly','staunchness','steadfast','steadfastly','steadfastness','steadiest','steadiness','steady','stellar','stellarly','stimulate','stimulates','stimulating','stimulative','stirringly','straighten','straightforward','streamlined','striking','strikingly','striving','strong','stronger','strongest','stunned','stunning','stunningly','stupendous','stupendously','sturdier','sturdy','stylish','stylishly','stylized','suave','suavely','sublime','subsidize','subsidized','subsidizes','subsidizing','substantive','succeed','succeeded','succeeding','succeeds','succes','success','successes','successful','successfully','suffice','sufficed','suffices','sufficiently','suitable','sumptuous','sumptuously','sumptuousness','super','superb','superbly','superior','superiority','supple','support','supported','supporter','supporting','supportive','supports','supremacy','supreme','supremely','supurb','supurbly','surmount','surpass','surreal','survival','survivor','sustainability','sustainable','swank','swankier','swankiest','swanky','sweeping','sweet','sweeten','sweetheart','sweetly','sweetness','swift','swiftness','talent','talented','talents','tantalize','tantalizing','tantalizingly','tempt','tempting','temptingly','tenacious','tenaciously','tenacity','tender','tenderly','terrific','terrifically','thank','thankful','thinner','thoughtful','thoughtfully','thoughtfulness','thrift','thrifty','thrill','thrilled','thrilling','thrillingly','thrills','thrive','thriving','thumb-up','thumbs-up','tickle','tidy','time-honored','timely','tingle','titillate','titillating','titillatingly','togetherness','tolerable','toll-free','top','top-notch','top-quality','topnotch','tops','tough','tougher','toughest','tranquil','tranquility','transparent','treasure','tremendously','trendy','triumph','triumphal','triumphant','triumphantly','trivially','trophy','trouble-free','trump','trumpet','trusted','trusting','trustingly','trustworthiness','trustworthy','trusty','truthful','truthfully','truthfulness','twinkly','ultra-crisp','unabashed','unabashedly','unaffected','unassailable','unbeatable','unbiased','unbound','uncomplicated','unconditional','undamaged','undaunted','understandable','undisputable','undisputably','undisputed','unencumbered','unequivocal','unequivocally','unfazed','unfettered','unforgettable','unity','unlimited','unmatched','unparalleled','unquestionable','unquestionably','unreal','unrestricted','unrivaled','unselfish','unwavering','upbeat','upgradable','upgradeable','upgraded','upheld','uphold','uplift','uplifting','upliftingly','upliftment','upscale','usable','useable','useful','user-friendly','user-replaceable','valiant','valiantly','valor','variety','venerate','verifiable','veritable','versatile','versatility','vibrant','vibrantly','victorious','victory','viewable','vigilance','vigilant','virtue','virtuous','virtuously','visionary','vivacious','vivid','vouch','vouchsafe','warmer','warmhearted','warmly','warmth','wealthy','welcome','well','well-backlit','well-balanced','well-behaved','well-being','well-bred','well-connected','well-educated','well-established','well-informed','well-intentioned','well-known','well-made','well-managed','well-mannered','well-positioned','well-received','well-regarded','well-rounded','well-run','well-wishers','wellbeing','whoa','wholeheartedly','wholesome','whooa','whoooa','wieldy','willing','willingly','willingness','windfall','winnable','winner','winners','winning','wins','wisdom','wise','wisely','witty','won','wonder','wonderful','wonderfully','wonderous','wonderously','wonders','wondrous','woo','workable','worked','works','world-famous','worth-while','worthwhile','wow','wowed','wowing','wows','yay','youthful','zeal','zenith','zest','zippy']
        sentenceSum = 0
        sentiment4OneDay4OneCom = []
        if len(twts) == 0:
            return [0.5,0.5,0.5,0.5]
        for sentence in twts:
            sentence = HTMLParser.HTMLParser().unescape(sentence)                            # unescape HTML
            sentence = re.sub(r"http\S+", "", sentence)                   # remove normal URLS
            sentence = re.sub(r"pic\.twitter\.com/\S+", "", sentence)     # remove pic.twitter.com URLS
            sentenceSum+=1
            wordsList = sentence.rstrip().split(' ')
            # print (sentenceList,"sentenceList\n")
            # print(sentence+'\n')
            for word in wordsList:
                word = word.lower() 
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
        sumAll = sentDim['Alert'] + sentDim['Happy'] + sentDim['Calm'] + sentDim['Kind']
        print ('\nalert happy calm kind')
        print (sentDim['Alert'],sentDim['Happy'],sentDim['Calm'],sentDim['Kind'])
        if sumAll == 0:
            return [0.5,0.5,0.5,0.5]
        sentiment4OneDay4OneCom.append(sentDim['Alert'] / float(sumAll) )   
        sentiment4OneDay4OneCom.append(sentDim['Happy'] / float(sumAll))
        sentiment4OneDay4OneCom.append(sentDim['Calm'] / float(sumAll))
        sentiment4OneDay4OneCom.append(sentDim['Kind'] / float(sumAll))
        print (sentiment4OneDay4OneCom,self.todayDate)
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
                 "My rank is very high, that's awesome",
                 "we moved legit two suburbs and u cut our internet off for 2 weeks??? Like wtf next time I'm switching like this is bs >:("]
        
        sentDim = {'neg':'Alert','pos':'Happy', 'neu':'Calm','compound':'compound'}
        
        sentiment4OneDay4OneCom = []
        sentimentAlert = []
        sentimentCalm = []
        sentimentHappy = []
        sentimentCompound = []

        sid = SentimentIntensityAnalyzer()
        assert(len(twts)>=0)
        if len(twts) == 0:
            return [0.5,0.5,0.5,0.5] #default
        for sentence in twts:
            sentence = HTMLParser.HTMLParser().unescape(sentence)                            # unescape HTML
            sentence = re.sub(r"http\S+", "", sentence)                   # remove normal URLS
            sentence = re.sub(r"pic\.twitter\.com/\S+", "", sentence)     # remove pic.twitter.com URLS
            # print(sentence+'\n')
            i = 0
            ss = sid.polarity_scores(sentence)
            for k in ss:
                # print('{0}: {1}, '.format(sentDim[k], ss[k]))
                if sentDim[k] == 'Alert':
                    sentimentAlert.append(ss[k])
                if sentDim[k] == 'Calm':
                    if ss[k] == 1:
                        sentimentCalm.append(0)
                    else:
                        sentimentCalm.append(ss[k])
                if sentDim[k] == 'compound':
                    sentimentCompound.append(ss[k])
                if sentDim[k] == 'Happy':
                    sentimentHappy.append(ss[k])
        print ('\nalert happy calm compound')
        sentiment4OneDay4OneCom.append(sum(sentimentAlert) / float(len(sentimentAlert)))    
        sentiment4OneDay4OneCom.append(sum(sentimentHappy) / float(len(sentimentHappy)))
        sentiment4OneDay4OneCom.append(sum(sentimentCalm) / float(len(sentimentCalm)))
        sentiment4OneDay4OneCom.append(sum(sentimentCompound) / float(len(sentimentCompound)))
        print (sentiment4OneDay4OneCom, self.todayDate)
        return sentiment4OneDay4OneCom

    def sentimentAnalyzerUsingIndicoio(self, twts):
        print ('loading....')
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
            sentence = HTMLParser.HTMLParser().unescape(sentence)                            # unescape HTML
            sentence = re.sub(r"http\S+", "", sentence)                   # remove normal URLS
            sentence = re.sub(r"pic\.twitter\.com/\S+", "", sentence)     # remove pic.twitter.com URLS
            # print(sentence+'\n')
            i = 0
            sums+=1
            ss = indicoio.emotion(sentence)
            # print (ss,sums, 'out of 1000')
            for k in ss:
                # print (k)
                # print('{0}: {1}, '.format(sentDim[k], ss[k]))
                if sentDim[k] == 'Alert':
                    sentimentAlert.append(ss[k])
                if sentDim[k] == 'Happy':
                    sentimentHappy.append(ss[k])
                if sentDim[k] == 'Sad':
                    sentimentSad.append(ss[k])
                if sentDim[k] == 'Surprise':
                    sentimentSurprise.append(ss[k])
        # print (sentimentHappy)
        print ('\nalert happy sad surprise')
        sentiment4OneDay4OneCom.append(sum(sentimentAlert) / float(len(sentimentAlert)))    
        sentiment4OneDay4OneCom.append(sum(sentimentHappy) / float(len(sentimentHappy)))
        sentiment4OneDay4OneCom.append(sum(sentimentSurprise) / float(len(sentimentSurprise)))
        sentiment4OneDay4OneCom.append(sum(sentimentSad) / float(len(sentimentSad)))
        print (sentiment4OneDay4OneCom, self.todayDate)
        return sentiment4OneDay4OneCom

if __name__ == '__main__':
    mode = sys.argv[1]
    assert(sys.argv[1]!='')
    sent = sentimentAnalyzerMultiDims()
    sent.mode = int(sys.argv[1])
    print ('Mode Selection: 0 for Vader, 1 for WordList, 2 for Indicoio(not recommended, cost too much time)')
    if len(sys.argv) < 3:
        print ('If you want to test, please give how many dates(less than 100) you want to analyze, otherwise we will run all the data\n')
    else:
        sent.days = int(sys.argv[2])
    sent.read()