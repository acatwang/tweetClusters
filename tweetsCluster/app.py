#! /usr/bin/env python
from sklearn.cluster import MiniBatchKMeans
import tweepy
from apiAuth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
import numpy as np
from MyTweet import MyTweet
from sklearn.feature_extraction.text import TfidfVectorizer
from util import STOPWORDS,process
MAX_TWEET = 1000

class App(object):
    """
    Take user-query and return clustered tweets
    """
    def __init__(self):
        self.api = self._getApi()
        self.tweets = []

    def _getApi(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

        return tweepy.API(auth)

    def getTweets(self,query,limit=MAX_TWEET):
        # TODO: solve time limit issue
        for t in tweepy.Cursor(self.api.search, q=query,lang='en').items(limit):
            #print t
            self.tweets.append(MyTweet(t))

        return self.tweets

    def getFeatures(self,tweets):
        # tfidf matrix
        rewroteText = [t.processedText for t in tweets]
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9,
                                           max_features=20000,
                                            min_df=0.1,
                                            stop_words=STOPWORDS,
                                            use_idf=True,
                                            tokenizer=process,
                                            ngram_range=(1,3))

        tfidfMatrix =tfidf_vectorizer.fit_transform(rewroteText)
        print "tfidf Martirx shape:{}",format(tfidfMatrix.shape)

        # others

        return tfidfMatrix

    def search(self,query,tweetLimit):
        print "Searching for the query: {0}".format(query)

        tweets = self.getTweets(query,tweetLimit)
        features = self.getFeatures(tweets)
        labels,centroid = self.kmean(features)
        for i, tag in enumerate(labels):
            print "{}: {}".format(tag,tweets[i].text.encode('ascii', 'ignore'))

    def kmean(self,data,K=3):
        # TODO: store clusters in an efficient data structure

        print np.shape(data)
        km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

        km.fit(data)

        print np.shape(km.cluster_centers_)
        print km.labels_
        return km.labels_,km.cluster_centers_.argsort()[:, ::-1]


app = App()
app.search('cat',30)