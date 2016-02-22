from util import readTweetObjects
from MyTweet import MyTweet
from app import App

tweetSamples = readTweetObjects('apple.json')
tweets = []
for t in tweetSamples:
    tweets.append(MyTweet(t))

app = App()
X = app.getFeatures(tweets)
clusters = app.clusterAndRank(X,3,tweets)
app.present(clusters,3)



