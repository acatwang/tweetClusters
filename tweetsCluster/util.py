from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string,json
from collections import namedtuple

from sklearn.feature_extraction.text import TfidfVectorizer



# Using NLTK toolset for text processing
STOPWORDS = stopwords.words() + list(string.punctuation) + ['rt']
regexTknzr = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

def process(rawTweet,tokenizer=regexTknzr,stemmer=p_stemmer):
	tweet = rawTweet
	tokens = tokenizer.tokenize(tweet)
	delStops = [s for s in tokens if s not in STOPWORDS]
	stemmed = [stemmer.stem(s) for s in delStops]
	return stemmed

def convertToObj(jsnDict,name="Tweet"):
    # name == 'entities':
     #   print jsnDict

    if all(not isinstance(v,dict) for v in jsnDict.values()):
       return namedtuple(name,jsnDict.keys())(*jsnDict.values())


    for k,v in jsnDict.items():
        #print type(v)
        if isinstance(v,dict):
            jsnDict[k] = convertToObj(v,k)

    return jsnDict.values()


def readTweetObjects(filepath):
    tweetSamples = []
    with open(filepath) as j:
        for line in j.readlines():
            jsonFile = json.loads(line)
            tweetObj = namedtuple('Tweet', jsonFile.keys())(*convertToObj(jsonFile))
            # Make each entity in entities a nametuple
            for f in tweetObj.entities._fields:
                lst = getattr(tweetObj.entities,f)
                #print f,lst
                tweetObj = tweetObj._replace(entities=tweetObj.entities._replace(**{f:[namedtuple(f,obj.keys())(*obj.values()) for obj in lst]}))
            #print tweetObj.entities
            tweetSamples.append(tweetObj)


        print "Loaded {} tweet objects".format(str(len(tweetSamples)))
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