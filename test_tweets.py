import unittest
from tweetsCluster.tweets import MyApi

"""Unit tests"""

class TweetsTest(unittest.TestCase):

    def getRawResults(self):
        assert len(MyApi.getRawTweets('apple',10)) == 10


if __name__ == '__main__':
    api = MyApi()
    results = api.getRawTweets('apple',1)
    print results
    unittest.main()

