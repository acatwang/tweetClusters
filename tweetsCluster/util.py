from nltk.tokenize import RegexpTokenizer,TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string,json
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Expand stopword list for tweets
# Ref: http://techland.time.com/2009/06/08/the-500-most-frequently-used-words-on-twitter/
STOPWORDS = stopwords.words() + list(string.punctuation) + \
            ['rt','r','lol','oh','ha','haha','thanks','bit.ly','post','time','go','retweet',"i'm"]


# Using NLTK toolset for text processing
regexTknzr = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

# Use the following tokenizer and stemmer to better handle tweet texts
twtTokenizer = TweetTokenizer()
snowballStemmer = SnowballStemmer("english")

def process(rawTweet,tokenizer=twtTokenizer,stemmer=snowballStemmer):
	tweet = rawTweet
	tokens = tokenizer.tokenize(tweet)
	delStops = [s for s in tokens if s not in STOPWORDS]
	stemmed = [stemmer.stem(s) for s in delStops]
	return stemmed

def convertToObj(jsnDict,name="Tweet"):
    if all(not isinstance(v,dict) for v in jsnDict.values()):
       return namedtuple(name,jsnDict.keys())(*jsnDict.values())


    for k,v in jsnDict.items():
        #print type(v)
        if isinstance(v,dict):
            jsnDict[k] = convertToObj(v,k)

    return jsnDict.values()


def readTweetObjects(filepath,nlimit=None):
    tweetSamples = []
    with open(filepath) as j:
        for i,line in enumerate(j.readlines()):
            jsonFile = json.loads(line)
            #tweetObj = namedtuple('Tweet', jsonFile.keys())(*convertToObj(jsonFile))
            tweetObj = namedtuple('Tweet', jsonFile.keys())(*jsonFile.values())

            # Make user an object
            tweetObj = tweetObj._replace(**{'user':namedtuple('user',jsonFile['user'].keys())(*jsonFile['user'].values()),
                                            'created_at':datetime.strptime(jsonFile['created_at'],"%a %b %d %H:%M:%S +0000 %Y")})


            tweetSamples.append(tweetObj)

            if nlimit and i == nlimit-1:
                break
            #print tweetObj.entities
        print "Loaded {} tweet objects from {}".format(len(tweetSamples),filepath)

    t = tweetSamples[0]
    #print 'readTweetObject'
    #print t.user.screen_name
    #print type(t.user.screen_name)
    return tweetSamples

def readTweetsSample(filepath,limit=None):
    listOfTweets = []
    print "Loading corpus"
    with open(filepath,'r') as f:
        for cnt,tweet in enumerate(f.readlines()):
            tweet = tweet.strip()
            listOfTweets.append(tweet)
            if limit and cnt == limit-1:
                break
    return listOfTweets