#! /usr/bin/env python
from sklearn.cluster import MiniBatchKMeans
import tweepy
from apiAuth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
import numpy as np

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
            self.tweets.append(MyTweet(t))

        return self.tweets

    def getFeatures(self,myTweetObjects):
        pass

    def search(self,query):
        print "Searching for the query: {0}".format(query)

        tweets = self.getTweets(query)
        features = self.getFeatures(tweets)
        clusters = self.kmean(features)

        return clusters

    def kmean(self,data,K=3):
        print np.shape(data)
        km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

        km.fit(data)

        print np.shape(km.cluster_centers_)
        print km.labels_


