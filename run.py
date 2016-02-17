#! /usr/bin/env python

import sys

from tweetsCluster.app import App


def main():
    query = 'apple' # default query
    if len(sys.argv) >1:
        query = sys.argv[1] # TODO: sanitize input
    app = App()
    return app.search(query)

if __name__ == '__main__':
    main()