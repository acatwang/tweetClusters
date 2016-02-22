from util import readTweetObjects
from MyTweet import MyTweet
from app import App

tweetSamples = readTweetObjects('apple.json')
tweets = []
cnt = 0
for t in tweetSamples:
    mytweet = MyTweet(t)
    tweets.append(MyTweet(t))
    cnt += 1

app = App()
X = app.getFeatures(tweets)
clusters = app.clusterAndRank(X,3,tweets,True)
app.present(clusters,3)



