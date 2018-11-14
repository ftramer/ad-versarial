#!/usr/bin/env python2
# encoding: utf-8
from distutils.version import LooseVersion
import argparse
from PIL import Image
import io
import sys
import traceback
import hashlib
import magic
import easylist
import pymongo
import time
import random
from bson.binary import Binary
import ghostery_matcher
import json
import typing
from mitmproxy import ctx
from mitmproxy import exceptions

import log
logger = log.getdebuglogger("proxy")

MAX_COLOR = 255
SVG_TPL  = """<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#{0:02x}{1:02x}{2:02x}"/></svg>"""
HTML_TPL = """<html><body bgcolor="#{0:02x}{1:02x}{2:02x}"></body></html>"""


def _mime_sniff(content):
    return magic.from_buffer(content)


class Monochrome:
    def __init__(self):
        pass

    def load(self, loader):
        self.set_options(loader)

    def set_options(self, loader):
        loader.add_option(
            name="database",
            typespec=str,
            default="",
            help="mongodb database to use")
        loader.add_option(
            name="collection",
            typespec=str,
            default="",
            help="mongodb collection to use")
        loader.add_option(
            name="rules",
            typespec=typing.Optional[str],
            default=None,
            help="Filter rules to use. If not provided, downloads the current list from easylist.com.")
        loader.add_option(
            name="ghostery",
            typespec=typing.Optional[str],
            default=None,
            help="Ghostery bugs.json file to use. If not provided, it will only use filter rules.")
        loader.add_option(
            name="others",
            typespec=bool,
            default=False,
            help="Replace svg and html, too")
        loader.add_option(
            name="clear_db",
            typespec=bool,
            default=False,
            help="Clears database before use. All data is lost")

    def configure(self, updates):
        logger.info("Configuring")
        # setup is done here because options are updated after load to passed values
        self.db_name = ctx.options.database
        self.db_col  = ctx.options.collection

        rules = ctx.options.rules
        if rules:
            logger.info('Loading easylist from file %s', rules)
            self.blocker = easylist.load_file(rules)
        else:
            logger.info('Loading easylist from web')
            self.blocker = easylist.load_easylist()

        ghostery_file = ctx.options.ghostery
        if ghostery_file:
            logger.info('Using Ghostery')
            with open(ghostery_file) as f:
                bugs = json.load(f)
            self.ghostery = ghostery_matcher.matcher(bugs)
        else:
            self.ghostery = None

        self.REPLACE_OTHERS = ctx.options.others
        logger.info('Replacing %s', 'others, too' if self.REPLACE_OTHERS else 'images only')

        logger.info("Connecting to database %s", self.db_name)
        self.mongo = pymongo.MongoClient()
        if ctx.options.clear_db:
            logger.info('Dropping database')
            self.mongo.drop_database(self.db_name)
        self.db = self.mongo[self.db_name]

    def running(self):
        logger.info("Running")

    def response(self, flow):
        try:
            if not self.blocker.should_block(flow.request.url):
                return
            if self.ghostery:
                if not self.ghostery.isBug(flow.request.url):
                    return
            flow.response.decode()
            url = flow.request.url

            if len(flow.response.content) == 0:
                logger.debug("Skipping %s because of length=0", flow.request.url)

            if "Content-Type" in flow.response.headers:
                mime = flow.response.headers["Content-Type"]
            else:
                # guess mime type if not given https://tools.ietf.org/html/rfc7231#section-3.1.1.5
                mime = _mime_sniff(flow.response.content)

            # random color
            color = (random.randrange(MAX_COLOR + 1),
                     random.randrange(MAX_COLOR + 1),
                     random.randrange(MAX_COLOR + 1))

            # if mime in _IMAGE_TYPES:
            if self.REPLACE_OTHERS:
                if mime and mime == "text/html":
                    flow.response.content = HTML_TPL.format(color[0], color[1], color[2])
                    self._db_insert(None, color, None, None, url)
                    logger.debug("(html) Replaced %s with %s", flow.request.url, color)
                    return

                if mime and (mime.startswith("application/svg+xml") or mime.startswith("image/svg+xml")):
                    # print flow.response.content
                    flow.response.content = SVG_TPL.format(color[0], color[1], color[2])
                    self._db_insert(None, color, None, None, url)
                    logger.debug("(svg) Replaced %s with %s", flow.request.url, color)
                    return

            if mime and mime.startswith("image/"):
                try:
                    # get width and height of original
                    orig_io = io.BytesIO(flow.response.content)
                    orig_img = Image.open(orig_io)
                    width, height = orig_img.size

                    # save original image
                    content = Binary(flow.response.content, 0)
                    self._db_insert(content, color, width, height, url)

                    # create monochrome image
                    repl_io = io.BytesIO()
                    repl_img = Image.new('RGB', (width, height), color=color)
                    repl_img.save(repl_io, format='PNG')

                    # replace picture
                    repl_io.seek(0)
                    flow.response.content = repl_io.read()
                    flow.response.headers["Content-Type"] = "image/png"
                    logger.debug("(image) Replaced %s with %s", flow.request.url, color)
                except Exception as e:
                    logger.error("Couldn't replace %s of length %s", flow.request.url, len(flow.response.content))
                    raise e
        except Exception as e:
            # many responses declared as images are not really images and PIL fails to open them, e.g., tracking pixels
            logger.exception(e)

    def done(self):
        logger.info("Shutting down")
        time.sleep(3)
        logger.info("Closing MongoClient")
        self.mongo.close()

    def _db_insert(self, content, color, width, height, url):
        doc = {
            "content": content,
            "color": color,
            "width": width,
            "height": height,
            "url": url
        }
        self.db[self.db_col].insert_one(doc)


addons = [Monochrome()]
