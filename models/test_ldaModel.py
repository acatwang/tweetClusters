from gensim import corpora, models
from util import process


corpusName = "cats.txt"

class Classifier(object):

	def __init__(self,model,dictionary):
		self.dictionary = corpora.Dictionary.load(dictionary)
		self.lda = models.ldamodel.LdaModel.load(model)
		self.bagOfWords = []
		self.corpus = []

	def readCorpus(self,corpusName):
		print "Loading corpus"
		with open(corpusName,'r') as f:
			for tweet in f.readlines():
				tweet = tweet.strip()
				#print tweet
				self.corpus.append(tweet)
				#print process(tweet)
				self.bagOfWords.append(process(tweet))
		return self.bagOfWords

	def classify(self):
		groups = dict((groupId,[]) for groupId in range(20))

		for i,bow in enumerate(self.bagOfWords):
			print bow
			doc_vec = self.dictionary.doc2bow(bow)
			groupProb = self.lda[doc_vec]
			argMaxGroupId = max(groupProb,key=lambda x:x[1])[0]

			try:
				groups[argMaxGroupId].append(self.corpus[i])
			except KeyError:
				print argMaxGroupId
		return groups


cats = Classifier('tweetsRegexMdl','tweetsRegex.dict')
cats.readCorpus('apple.txt')
groups = cats.classify()
for group, tweets in groups.items():
	print "Group {}".format(group)
	if tweets:
		for tweet in tweets:
			print "- " +tweet
		print
