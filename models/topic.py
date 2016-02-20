from util import readTweetsSample,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import requests
from sklearn.cluster import KMeans, MiniBatchKMeans

"""
This an attempt to use Tagme Api to assign topic for each tweet
Did not work out because the outcome is bad
Probably because the API is based on Wikipedia corpus, which is essentially different from Tweets
"""
TOPIC_THRESHOLD = 0.1
MAX_TOPIC = 10

tweets = readTweetsSample('/Users/yywang/Github/tweetsCluster/apple.txt',500)
print tweets[:10]
bagOfTopics = []
def tag(tweet):
    if not tweet:
        return [0]*10
    TagmeAPI = "http://tagme.di.unipi.it/tag?key=pqqpqqh452zz910&text={}".format(tweet)
    resp = requests.get(TagmeAPI)

    topics = []
    if resp.status_code != 200: # Something went wrong.
        print resp.status_code
        print resp.text

    for topic in resp.json().get('annotations'):
        if topic['rho'] > TOPIC_THRESHOLD:
            topics.append(topic.get('title'))
    return topics
"""
for t in tweets:
    bagOfTopics.append(tag(t))
    #print t
"""


def myAnalyzer(doc):
    """
    Overwrite the build-in analyzer
    :param doc: bag of topics
    :return:bag of topics
    """
    return doc

vectorizer = CountVectorizer(analyzer="word",
                             max_features=MAX_TOPIC,
                             stop_words=STOPWORDS,
                             tokenizer=None,
                             preprocessor = None)
X = vectorizer.fit_transform(tweets)
topicName = vectorizer.get_feature_names()

def kmean(data,K=3):

    print np.shape(data)
    km = MiniBatchKMeans(n_clusters=K, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)

    km.fit(data)

    return km.labels_,km.cluster_centers_.argsort()[:, ::-1]



labels,centroid = kmean(X)
for i in range(3):
    print "Cluster %d words:" % i
    for ind in centroid[i,:2]:
        print topicName[ind]


for i, tag in enumerate(labels):
    print "{}: {}".format(tag,tweets[i])
