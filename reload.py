import argparse
import time
import os
import logging
import requests

def reload(timestamp, ep):
    url = "http://localhost:8080/creep_control/load"

    payload = "{\n  \"ep\": %s,\n  \"timestamp\": %s\n}" % (timestamp, ep)
    headers = {
        'Content-Type': "application/json",
        'Cache-Control': "no-cache",
        }
    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--timestamp', type=str, required=True)
    parser.add_argument('--ep', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    reload(timestamp=args.timestamp, ep=args.ep)
