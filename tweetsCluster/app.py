#! /usr/bin/env python
from sklearn.cluster import MiniBatchKMeans
import tweepy
from apiAuth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
import numpy as np
from MyTweet import MyTweet
from sklearn.feature_extraction.text import TfidfVectorizer
from util import STOPWORDS,process
import json, datetime


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

    def getTweets(self,query,limit=MAX_TWEET,store=False):
        # TODO: solve time limit issue
        tweetCnt = 0
        for statusObj in tweepy.Cursor(self.api.search, q=query,lang='en').items(limit):
            #print statusObj
            tweetCnt +=1
            self.tweets.append(MyTweet(statusObj))
            if store:
                with open(query+".json","a") as f:
                    f.write(json.dumps(statusObj._json)+"\n")
        print "Got total {} tweets".format(tweetCnt)
        return self.tweets

    def getFeatures(self,tweets):
        # tfidf matrix
        rewroteText = [t.processedText for t in tweets]
        tfidf_vectorizer = TfidfVectorizer(max_df=.8,
                                           max_features=10000,
                                            min_df=0.001,
                                            stop_words=STOPWORDS,
                                            use_idf=True,
                                            tokenizer=process,
                                            ngram_range=(1,3))

        tfidfMatrix =tfidf_vectorizer.fit_transform(rewroteText)
        print "tfidf Martirx shape:{}",format(tfidfMatrix.shape)

        # others
        self.tfidfDict = tfidf_vectorizer.get_feature_names()
        return tfidfMatrix

    def search(self,query,K=5,tweetLimit=MAX_TWEET,store=False):
        print "Searching for the query: {0}".format(query)

        tweets = self.getTweets(query,tweetLimit,store)
        features = self.getFeatures(tweets)

        labels,centers_features = self.kmean(features,K)
        clusters = self.rankInCluster(labels,centers_features,K)

        for i in range(K):
            print "Cluster {}: {}".format(i, clusters[i]['words'])
            print "Best Tweet: {}".format(clusters[i]['best'])
            print "First Tweet: {}".format(clusters[i]['first'])
            for twt in clusters[i]['all']:
                print twt.text.encode('ascii', 'ignore')
            print
            print

    def kmean(self,data,K=5):
        # TODO: store clusters in an efficient data structure

        print np.shape(data)
        km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

        km.fit(data)

        print np.shape(km.cluster_centers_)
        print km.labels_
        return km.labels_,km.cluster_centers_

    def rankInCluster(self,labels,centers_features,K):
        clusters = dict((clusId,{'all':[],'best':"",'first':"",'words':""}) for clusId in range(K))

        # In each cluster, sort tweets by time
        for i,label in enumerate(labels):
            clusters[label]['all'].append(self.tweets[i])
        for label in labels:
            clusters[label]['all'] = sorted(clusters[label]['all'], key=lambda x:x.time)
            clusters[label]['first'] = "{0} ({1})".format(clusters[label]['all'][0].text.encode('utf-8', 'ignore'),
                                                          datetime.datetime.fromtimestamp(clusters[label]['all'][0].time).strftime('%Y-%m-%d %H:%M:%S'))
        #TODO: find best tweet in cluster

        # Get the top words in each cluster
        sorted_centers_features = centers_features.argsort()[:, ::-1]
        for ctr in xrange(K):
            top3words = []
            for field in sorted_centers_features[ctr,:3]: # Get the top 3 common words
                try:
                    top3words.append(self.tfidfDict[field].encode('utf-8', 'ignore'))
                except IndexError:
                    continue
            clusters[ctr]['words'] = "/".join(top3words)

        return clusters
app = App()
app.search('single',3,100, False)