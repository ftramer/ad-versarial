#!/usr/bin/env python3

import argparse
import logging
import hashlib
import sys
import subprocess
import misc.bounding_boxes as bounding_boxes
import os.path
import glob
import urllib3
import time

LOGGER_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=LOGGER_FORMAT,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
# increase this number if proxy needs too long to start
PROXY_STARTUP = 8.


def main():
    args = parse_args()
    db = args.database
    ro = args.replace_others
    el = args.easylist
    gh = args.ghostery
    port = args.port
    clear = args.clear_database

    with open(args.file) as file:
        for line in file.readlines():
            url = line.split(',')[0].strip()
            col = hashlib.sha256(url.encode()).hexdigest()
            # start proxy
            proxy = start_proxy(port, db, col, ro, el, gh, clear)
            time.sleep(PROXY_STARTUP)
            # start fridolin
            frido = start_fridolin(url, "localhost:{}".format(port))
            # stop proxy
            success = False
            try:
                frido.wait()
                success = frido.returncode == 0
                proxy.terminate()
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.error("Proxy did not terminate fast enough")
                proxy.kill()
            if not success:
                logging.error("Skip stitching for %s", url)
                continue

            # get screenshot path
            host = urllib3.util.parse_url(url).host
            path = os.path.join("data", host, '*', 'main.png')
            folders = glob.glob(path)
            if (len(folders) == 0):
                logging.error("Skip stitching for %s", url)
                continue
            path = max(folders, key=os.path.getctime)
            # stitch image
            start_stitcher(db, col, path)


def start_proxy(port, database, collection, replace_others, easylist, ghostery, clear):
    script = "proxy/mitm-colored-ads.py"
    cmd = [
        "mitmdump",
        "--listen-port", str(port),
        "--quiet", "--no-http2", "--anticache",
        "-s", script,
        "--set", "database={}".format(database),
        "--set", "collection={}".format(collection),
        "--set", "rules={}".format(easylist),
        "--set", "ghostery={}".format(ghostery),
        "--set", "other={}".format("true" if replace_others else "false"),
        "--set", "clear_db={}".format("true" if clear else "false")]
    return subprocess.Popen(cmd)


def start_fridolin(url, proxy):
    cmd = ["python2", "fridolin/fridolin.py", url, proxy]

    return subprocess.Popen(cmd)


def start_stitcher(database, collection, path):
    bounding_boxes.compute_boxes(database, collection, path)


def parse_args():
    parser = argparse.ArgumentParser(description='Crawls a list of urls and screenshots the websites')
    parser.add_argument('file', help='List of URLs')
    parser.add_argument('database',
        help='Specifies the database to be used')
    parser.add_argument('--clear-database', action='store_true',
        help='Clears the database before use')
    parser.add_argument('--replace-others', action='store_true',
        help='Replace not only image ads')
    parser.add_argument('--ghostery',
        help='ghostery database to be used')
    parser.add_argument('--easylist',
        help='easylist to use')
    parser.add_argument('--port', default=8080,
        help='Proxy port')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
