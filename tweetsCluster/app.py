#! /usr/bin/env python
from sklearn.cluster import MiniBatchKMeans
import tweepy
from apiAuth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
import numpy as np
from MyTweet import MyTweet
from sklearn.feature_extraction.text import TfidfVectorizer
from util import STOPWORDS,process
import json, datetime
from scipy.spatial.distance import euclidean, pdist
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize,scale
from sklearn import metrics


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

    def sayHi(selfs):
        print "Hi there"

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
        tfidf_vectorizer = TfidfVectorizer(max_df=.5,
                                           max_features=5000,
                                            min_df=0.05,
                                            stop_words=STOPWORDS,
                                            use_idf=True,
                                            tokenizer=process,
                                            ngram_range=(1,3))

        tfidfMatrix =tfidf_vectorizer.fit_transform(rewroteText)
        print "tfidf Martirx shape:{}",format(tfidfMatrix.shape)

        self.tfidfDict = tfidf_vectorizer.get_feature_names()
        print self.tfidfDict
        context = []
        for t in tweets:
            context.append([t.timezone,t.time,t.hasPhoto,t.sentimentScore])
        context = np.array(context)

        #print "tdidf {}".format(tfidfMatrix.shape)
        #print "norm {} {}".format(normContext.shape,type(normContext))
        features = np.hstack((tfidfMatrix.A,context))
        print features.shape
        return features

        #print features.shape
        #return features

    def search(self,query,K=3,tweetLimit=MAX_TWEET,store=False):
        print "Searching for the query: {0}".format(query)

        tweets = self.getTweets(query,tweetLimit,store)
        tweet2features = self.getFeatures(tweets)
        clusters = self.clusterAndRank(tweet2features,K)

        self.present(clusters,K)


    def present(self,clusters,K):
        for i in range(K):
            print "Cluster {}: {}".format(i, clusters[i]['words'])
            print "Best Tweet: {}".format(clusters[i]['best'])
            print "First Tweet: {}".format(clusters[i]['first'])
            for twt in clusters[i]['all'][:10]:
                print "\t - " + twt.printTweet()
            print
            print


    def clusterAndRank(self,X,K=3,tweets=None,diagnose=False):
        #print np.shape(X)
        km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)
        km.fit(X)
        if diagnose:
            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(X, km.labels_, sample_size=1000))
            print()

        clusters = self.rankInCluster(km.labels_,km.cluster_centers_,K,X,tweets)

        return clusters

    def rankInCluster(self,labels,centers_features,K,X,tweets=None):
        clusters = dict((clusId,{'all':[],'best':"",'first':"",'words':"","n":0}) for clusId in range(K))
        if not tweets:
            tweets = self.tweets
        # In each cluster, do the following :
        # 1) sort tweets by created time in descending order
        # 2) get the first tweet (in terms of time)
        # 3) find the tweet that is closet to the cluster centroid (best tweet)

        for i,label in enumerate(labels):
            clusters[label]['all'].append(tweets[i])
            clusters[label]['n'] += 1

        for label in labels:
            clusters[label]['all'] = sorted(clusters[label]['all'], key=lambda x:x.time, reverse=True)
            clusters[label]['first'] = clusters[label]['all'][-1].printTweet()

        # Find the best tweet in each cluster
        for clusId in xrange(K):
            print "{} tweets in cluster {}".format(len(clusters[clusId]['all']), clusId)
            tweetIdxInClus = np.where(labels == clusId)
            if not clusters[clusId]["n"]:
                break
            #print tweetIdxInClus
            centerCoord = centers_features[clusId].reshape(1,-1)
            distToCtr = pairwise_distances(X[tweetIdxInClus], centerCoord)  # dimension: (n_tweets, 1)


            # Calculate tweet popularity/quality feature
            popularity = []
            for i,t in enumerate(tweets):
                if i in tweetIdxInClus[0]:
                    popularity.append([t.retweetCnt,t.favCnt,t.isRetweet,t.followers])
            popularity = np.array(popularity)  # n_tweet X 5
            coef = np.array([.5,.5,-.8,.2]) # hard-coded coefficient
            #print "popularity:{}".format(popularity.dot(coef).shape)
            norm_popularity = normalize(popularity).dot(coef).reshape(-1,1)
            #print norm_popularity
            #print norm_popularity.shape

            feat = np.add(distToCtr, norm_popularity)
            bestTweetId = np.argmax(feat)
            clusters[clusId]['best'] = tweets[tweetIdxInClus[0][bestTweetId]].printTweet()

        # Get the top words in each cluster
        sorted_centers_features = centers_features.argsort()[:, ::-1]
        for ctr in xrange(K):
            top3words = []
            for field in sorted_centers_features[ctr,:5]: # Get the top 3 common words
                try:
                    top3words.append(self.tfidfDict[field].encode('utf-8', 'ignore'))
                except IndexError:
                    continue
            clusters[ctr]['words'] = "/".join(top3words)
        return clusters
