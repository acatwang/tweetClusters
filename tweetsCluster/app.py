#! /usr/bin/env python

from tweetsCluster.tweets import MyApi


class App(object):
    def __init__(self):
        self.api = MyApi()

    def search(self,query):
        clusters = self.getCluster(query)
        print "Searching for the query: {0}".format(query)
        output = []
        for i,c in enumerate(clusters):
            output.append('Tweet number {0}:{1}'.format(i,c))
            return output

    def getCluster(self,query):
        clusters = {}

        tweets = self.api.getTweets(query,10)

        return tweets
