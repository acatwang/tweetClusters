#!/bin/bash

echo `dirname $0`/..

#set up environment
source venv/bin/activate
pip install -r requirements.txt

# Take parameter from input
query="apple"
k=3
nTweet=1000

usage(){
    echo 'Usage: ./tweetCluster.sh -q <query> -k <numCluster> -n <tweetLimitNum>'
    exit
}

# http://wiki.bash-hackers.org/howto/getopts_tutorial

options_found=0

while getopts ":q:k:n:" opt; do
    options_found=1
    case "$opt" in
        q)	query=$OPTARG
		;;
	k)	k=$OPTARG
		;;
	n)	nTweet=$OPTARG
		;;
	:)	echo "Option -$OPTARG requires an argument." >&2
      		exit 1
		;;
        \? ) 	
		usage
		exit 1
		;; 
    esac
done

#shift "$((OPTIND-1))" # Shift off the options and optional --.
 
if ((!option_found && $#==1)); then
    query=$1
else 
    echo "Please enter query"
fi

echo "query: $query, numOfgroup:$k, tweetNum:$nTweet"
python run.py query k nTweet
