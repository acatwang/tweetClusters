#! /usr/bin/env python

import sys

from tweetsCluster.app import App


def main():
    #print sys.argv
    if len(sys.argv) >1:
        query = sys.argv[1] # TODO: sanitize input
        try:
            k = int(sys.argv[2])
            tweetlimit = int(sys.argv[3])
        except ValueError:
            k = 3
            tweetlimit = 1000


    app = App()
    app.sayHi()
    app.search(query,k,tweetlimit,False)

if __name__ == '__main__':
    main()