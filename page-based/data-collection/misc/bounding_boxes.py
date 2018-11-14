#!/usr/bin/env python3

import logging
import argparse
from PIL import Image
import sys
import pymongo
import io
import os
import json

LOGGER_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=LOGGER_FORMAT,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

THRESHOLD_WIDTH = 20
THRESHOLD_HEIGHT = 20

class Box(object):
    def __init__(self, maxX, maxY, width, height, orig):
        self.top = maxY
        self.left = maxX
        self.bottom = 0
        self.right = 0
        self.width = width
        self.height = height
        self.original = orig

    def update(self, coords):
        x, y = coords
        self.top = min(y, self.top)
        self.left = min(x, self.left)
        self.bottom = max(y, self.bottom)
        self.right = max(x, self.right)

    def fits_height(self):
        return self.get_height == self.height

    def get_height(self):
        return self.bottom - self.top + 1

    def fits_width(self):
        return self.get_width == self.width

    def get_width(self):
        return self.right - self.left + 1

    def fits(self):
        return self.fits_height() and self.fits_height()

    def was_updated(self):
        return self.bottom - self.top >= 0 and self.right - self.left >= 0

    def __str__(self):
        s = 't{} l{} b{} r{}'. format(self.top, self.left, self.bottom, self.right)
        s += ' h is/should {}/{}'.format(self.bottom - self.top + 1, self.height)
        s += ' w is/should {}/{}'.format(self.right - self.left + 1, self.width)
        return s

    def __dict__(self):
        d = {"top": self.top,
             "left": self.left,
             "bottom": self.bottom,
             "right": self.right}
        return d

    def __repr__(self):
        return self.__str__()


def stitch(database, collection, path):
    db_name = database
    col = collection
    file = path

    with pymongo.MongoClient() as mongo:
        db = mongo[db_name]

        logging.info("Opening %s", file)
        im = Image.open(file)
        rgb_im = im.convert('RGBA')
        height = im.height
        width = im.width

        logging.info("Setting up boxes")
        boxes = setup_boxes(db, col, width, height)
        logging.info("Scanning")
        scan(rgb_im, boxes)
        logging.info("Replacing")
        for color, box in boxes.items():
            replace(rgb_im, box, color)
        #rgb_im.show()
        new_path = os.path.join(os.path.dirname(path), "main-withad.png")
        rgb_im.save(new_path)


def compute_boxes(database, collection, path):
    db_name = database
    col = collection
    file = path

    with pymongo.MongoClient() as mongo:
        db = mongo[db_name]

        logging.info("Opening %s", file)
        im = Image.open(file)
        rgb_im = im.convert('RGBA')
        height = im.height
        width = im.width

        logging.info("Setting up boxes")
        boxes = setup_boxes(db, col, width, height)
        logging.info("Scanning")
        scan(rgb_im, boxes)
        # filter invisible boxes
        boxes = {str(k): v for k, v in boxes.items() if v.was_updated()}
        json_out = os.path.join(os.path.dirname(path), "main-boxes.json")
        with open(json_out, 'w+') as j:
            j.write(json.dumps(boxes, default=lambda x: x.__dict__(), indent=2, sort_keys=True))


def scan(im, boxes):
    '''Searches the image for the known colors. This needs some tweaking. If 
    the monochrom image was resized it is often no longer monochrome so boxes 
    are not detected completely.'''
    height = im.height
    width = im.width
    x, y = 0, 0
    while y < height:
        x = 0
        while x < width:
            color = im.getpixel((x, y))
            if color in boxes:
                b = boxes[color]
                count_right = scan_right(im, (x, y), color)
                if count_right > THRESHOLD_WIDTH:
                    b.update((x, y))
                    x += count_right - 1
                    b.update((x, y))
            x += 1
        y += 1


def scan_right(im, coords, color):
    count = 0
    x, y = coords
    while im.getpixel((x, y)) == color and x < im.width:
        count += 1
        x += 1
    return count


def scan_down(im, coords, color):
    count = 0
    x, y = coords
    while im.getpixel((x, y)) == color and y < im.height:
        count += 1
        y += 1
    return count


def replace(im, box, color):
    if box.was_updated():
        logging.info("Replacing started at %s/%s to %s/%s", box.left, box.top, box.right, box.bottom)
        bio = io.BytesIO(box.original)
        bimg = Image.open(bio)
        bimg = bimg.convert("RGBA")
        replaced = 0
        height = min(box.height, box.get_height())
        width = min(box.width, box.get_width())
        total = height * width
        for y in range(height):
            for x in range(width):
                pixel = ((box.left + x, box.top + y))
                if im.getpixel(pixel) == color:
                    replaced += 1
                    im.putpixel(pixel, bimg.getpixel((x, y)))
                else:
                    print(pixel)
        logging.info("Replaced %s of %s (%s%%)", replaced, total, replaced / total * 100)


def setup_boxes(db, col, maxX, maxY):
    boxes = {}
    for record in db[col].find():
        color = tuple(record['color'] + [255])
        width = record['width']
        height = record['height']
        orig = record['content']
        boxes[color] = Box(maxX, maxY, width, height, orig)
    return boxes


def rgb_to_hex(r,g,b):
    return '{:02x}{:02x}{:02x}'.format(r, g, b)


def parse_args():
    parser = argparse.ArgumentParser(description='Tool to detect bounding boxes')
    parser.add_argument('--database', '-d', required=True, help='database to query')
    parser.add_argument('--collection', '-c', required=True, help='collection in database to get boxes')
    parser.add_argument('--path', '-p', required=True, help='image to scan')

    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == "__main__":
    args = parse_args()
    stitch(args.database, args.collection, args.path)
