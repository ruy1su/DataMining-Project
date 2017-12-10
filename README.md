# CS145DataMining-Project
Twitter project with topic: Stock Prediction 

# Team 
Don Lee, Shunji Zhan, Yaying Ye, Zeyu Zhang, Zixia Weng

# Install
    pip install -r requirements.txt
    pip3 install -r requirements.txt

# Run TweetsCrawler
This may take a while... results are saved in data-set/tweets/
    cd utils/
    python2 tweetsCrawler.py [since] [until]
    python2 tweetsCrawler.py 2010-01-01 2010-01-10

# Run Sentimental Analysis
    python2 sentimentalAnalyzerNp.py [mode (0,1)] [numDays (optional)]
    python2 sentimentalAnalyzerNp.py 0
    python2 sentimentalAnalyzerNp.py 1 10
    
    python2 sentimentalAnalyzerMulti.py [mode (0,1,2)] [numDays (optional)]
    python2 sentimentalAnalyzerMulti.py 0
    python2 sentimentalAnalyzerMulti.py 1 10
# Run Models and Evaluations
    python3 main.py

