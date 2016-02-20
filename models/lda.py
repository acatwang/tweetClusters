import string
import nltk.corpus
#nltk.download("twitter_samples")
#nltk.download("stop_words")
#nltk.download('webtext')
from nltk.tokenize import TweetTokenizer,RegexpTokenizer
from nltk.corpus import twitter_samples, stopwords,reuters
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from ..util import process

NUM_TOPIC = 10

# This file use twitter_samples corpus in nltk to train the LDA model
tweetTknzr = TweetTokenizer(strip_handles=True, reduce_len=True)


rawTweets = twitter_samples.strings('tweets.20150430-223406.json')
#reuters = [ reuters.raw(f) for f in reuters.fileids()]

#rawCorpus = rawTweets + reuters
rawCorpus = rawTweets

texts = [process(line,tweetTknzr) for line in rawCorpus]

print texts[:3]


dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]   # convert tokenized documents into a document-term matrix

dictionary.save('tweetsRegex.dict')
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPIC, id2word = dictionary, passes=20)
ldamodel.print_topics(num_topics=NUM_TOPIC, num_words=10)

ldamodel.save('tweetsRegexMdl')