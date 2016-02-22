from MyTweet import MyTweet
from util import readTweetObjects

tweetSamples = readTweetObjects('tweetObjects.json')


for tweet in tweetSamples:
    myTweet = MyTweet(tweet)
    #print myTweet.parseUrl(tweet.entities)
    #print myTweet.isRetweet
    print "{}: {}".format(myTweet.timezone, tweet.user.time_zone)

    print tweet.text
    print myTweet.processedText
    print