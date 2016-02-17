import unittest
from ..tweetsCluster.tweets import MyApi
"""
Unit tests

"""

class TweetsTest(unittest.TestCase):
    def test_getThreeTweets(self):
        api = MyApi()
        self.assertEqual(len(api.getTweets('apple',3)), 3)


if __name__ == '__main__':

    unittest.main()

