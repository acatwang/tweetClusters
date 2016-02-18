import string
import nltk.corpus
#nltk.download("twitter_samples")
#nltk.download("stop_words")
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples, stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models


# This file use twitter_samples corpus in nltk to train the LDA model
NUM_TOPIC = 12
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
stops = stopwords.words('english') + list(string.punctuation) + ['rt']
p_stemmer = PorterStemmer()


# Process twitter_samples corpus
rawTweets = twitter_samples.strings('tweets.20150430-223406.json')
texts = []

for rawTweet in rawTweets:
    tweet = rawTweet.lower()
    tokens = tknzr.tokenize(tweet)
    delStops = [s for s in tokens if s not in stops]
    stemmed = [p_stemmer.stem(s) for s in delStops]
    texts.append(stemmed)

#print texts[:3]



dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]   # convert tokenized documents into a document-term matrix

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPIC, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=5))

ldamodel.save('ldamodel')