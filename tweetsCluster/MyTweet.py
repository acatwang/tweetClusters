from urlparse import urlparse
import time,re,datetime
from collections import namedtuple


class MyTweet(object):
    """
    Parse and preprocess raw tweet object

    Reference:
    Tweets Object https://goo.gl/d1hExF
    Entity Object https://goo.gl/Sj7TuJ

    """
    def __init__(self,tweet):
        self.text = tweet.text
        self.creator = tweet.user.screen_name.encode('utf-8', 'ignore'),
        self.processedText = self.rewrite(tweet.text,
                                     tweet.user.screen_name,
                                     tweet.in_reply_to_user_id_str,
                                     tweet.in_reply_to_status_id)
        self.lang = tweet.lang
        self.domains, self.urlpPaths, self.hasPhoto = self.parseUrl(tweet.entities)
        self.timezone = tweet.user.utc_offset if tweet.user.utc_offset else 0
        self.sentimentScore = self.getScore(tweet.text)

        # For "best" results
        self.time = time.mktime(tweet.created_at.timetuple())
        self.retweetCnt = tweet.retweet_count + 0.0
        self.isRetweet = self.isRetweet(tweet.text) +0.0
        self.favCnt = tweet.user.favourites_count+0.0
        self.followers = tweet.user.followers_count + 0.0

    def rewrite(self,rawTweet, creatorName, *args):
        rawTweet = rawTweet.lower()

        # remove url and media link
        text = re.sub(r"http\S+", "", rawTweet)

        text += "@{} ".format(creatorName)
        for arg in args:
            if arg:
                text += "{} ".format(str(arg))
        return text

    def parseUrl(self,entities):
        """
        Unzip the url and parse it to extract features
        """
        domains, urlPaths = [], []
        hasPhoto = 0


        # urls
        for item in entities.get('urls'):
            try:
                url = item['expanded_url'] if item.get('expanded_url') else item['url']

            except AttributeError:
                continue
            parsed = urlparse(url)
            domains.append(parsed.netloc)
            urlPaths.append(parsed.path.strip("/"))

        # medias
        if entities.get('media'):
            for item in entities['media']:
                try:
                    hasPhoto = item['type'] == "photo"
                except AttributeError:
                    continue
                mediaUrl = item.get('media_url')
                parsed = urlparse(mediaUrl)
                domains.append(parsed.netloc)

        return domains, urlPaths, hasPhoto

    def isRetweet(self,tweet):
        return 'rt' in tweet.lower()

    def getScore(self,tweet):
        # sentiment analysis
        return 0

    def printTweet(self):
        #print self.creator
        return "{0} tweets at {1}: {2}".format(self.creator[0], # TODO:checkout why is it a tuple
                                               datetime.datetime.fromtimestamp(self.time).strftime('%Y-%m-%d %H:%M:%S'),
                                               self.text.encode('utf-8', 'ignore'))