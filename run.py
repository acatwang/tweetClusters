#! /usr/bin/env python

import sys
from app import App

def main():
    print sys.argv
    query = sys.argv[1] # TODO: sanitize input
    app = App()
    return app.search(query)

if __name__ == '__main__':

    main()