#! /usr/bin/env python

import tweets

class App(object):

    def search(self,query):
        clusters = self.getCluster(query)
        print "Searching for the query: {0}".format(query)
        print clusters

    def getCluster(self,query):
        return tweets.cluster
