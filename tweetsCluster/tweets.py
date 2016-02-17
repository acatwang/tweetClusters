#! /usr/bin/env python
import tweepy
from apiAuth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

MAX_TWEET = 1000

class MyApi(object):
    """
    reference :https://dev.twitter.com/rest/public/search
    """

    def __init__(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

        self.api = tweepy.API(auth)

    def getRawFields(self,tweet):
        pass

    def getTweets(self,query,limit=MAX_TWEET):
        searched_tweets = [ r for r in tweepy.Cursor(self.api.search, q=query,lang='en').items(limit)]
        cleanTweets = [r.text for r in searched_tweets]
        return cleanTweets

