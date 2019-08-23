#! /usr/bin/env python3

from skimage import draw, transform, util, filters, color, io
import math
from io import BytesIO
import os
import argparse
import hashlib
import json
import random
import numpy as np
import threading
import concurrent.futures
import sys

import logging
logger = logging.getLogger('GENERATOR')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# fh = logging.FileHandler("generator.log")
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

# directories to store files
IMAGES_PATH = 'images'
ANNOTATION_PATH = 'labels'
# Workers to use for initial ad analysis
WORKERS=2
# how many ads are considered for insertion
CANDIDATE_NEIGHBORS = 50
# how often an ad can be reused
REUSES = 20

# Images that do not match the width will be filled with background images left and right
WIDTH_SHOULD = 1919
HEIGHT_SHOULD = 1013

# number of images you want to create at most.
MAX_NUM_ADFREE = 100
MAX_NUM_BACKGROUND = 100
MAX_NUM_INTERSTITIAL = 100
MAX_NUM_ADCHOICE = 100
MAX_NUM_AD = 100
num_adfree = 0
num_background = 0
num_interstitial = 0
num_adchoice = 0
num_ad = 0

hori = 'horizontal'
vert = 'vertical'
SCREENSHOT_NAME = 'main.png'
BOX_NAME = 'main-boxes.json'

# Note: ratio is always width / height

# logos to place on ads
MISC = {
    'ac_icon': util.img_as_float(io.imread(BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x0f\x00\x00\x00\x0f\x08\x02\x00\x00\x00\xb4\xb4\x02\x1d\x00\x00\x01\x8eIDAT(\xcfmRK/\x03Q\x14\xfe\xee54*\rZ\xafi\x11Q}\xc4+6\xf6X\t?A\x13\t\t\x16bcgcm\xa3\x16\x12\x1bA$]Hl\x89\x85\xc4\x82\x04\x0bBI\x9a\xb6\x92\x8a\xa8\x91V=S:t\xee\xb5\x98\xceL[\xce\xe2\xcb\xbd\'\xdf9\xe7;\x0f\xc2\x15\x0e\n0\x80"g\xea;\x1f5?U?\xeb\xd2\xb3?\x96d\xaa\x97\x16"\xd3\xd8\x14\x84s\x1e\xfe\x94\xbd\xbb\x97\xa0\x80\xa5|\xa7\xa7e\xc8f)\xaec\xf0\x81\xc4w\x16\x14\xb0Z\xda\xaa\xcc\xc3\x87!r\x1c\xbe~\xcf\x14S\x99Q\x0f`\x08x\xc4h\xaf\xf3d\xa0\x13\xb2\xd2\xb5\x1f\xec?\x8b%\xb2JN\x86\xd6\x83`\xd4`\x1c\xc0\xd1kz\xc3+\xca\x8cM\x06\xef\xea\xe3\xa9i\xb7}\xc1-\x9a)Q94/\x9a\xa8\x81\xa9Lv\xa2\xd1\x96\x1e\xec\x99r\xd9\x97C\xf1\x8a\xbd\x8b\xcd\x87\x97\x82\x99\xe8\x8a\x82i\xf9*\xf3\r\xc0L\xc9\x8a\xc7~\xd0\xd7\x8e\xaf\x9f\xd1\xd3\xe8[\x96\x81\xeaJ\xb41m\xa53\xb5\x0c\x00\x9e\x7f\x94\x99\x88\x14\x08K\xa8([\xebh\xae\x14\xa8\xa6\xbbpL\xf7\x9c\xcfF\xa4\xc5\xf0\x03\x08\x99\xebn\x9aw\xd6\x97\x91\x9cnA\xef\x97\x03`\xb0\x02\x92\xf4\xb2\x98x\x1bs\x89K\x1e\xd1B(\x88\x91N\xd0\xd7\xeb\x8bJ\xbe\x9b\xc7\xd5\xd6\xbam\x93\xb0\xeev4\x94\x97\xfe=\x04\x01\x0c5\xa6\x120\xe0\xe9\xc3\xed\xb0\x8d\x88\xd5\xe3\xa2\xb5\xf8Nt\xe4\x9cs\xce\xfd\xb7\xc9\xf3\xf7O\xae\x9b\xf2?\x92\x7fn0\xdf\n\x97\xff\x0b/g\xc8t\xbb\xec\xd2m\x00\x00\x00\x00IEND\xaeB`\x82'))),
    'ac_x': util.img_as_float(io.imread(BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x0f\x00\x00\x00\x0f\x08\x02\x00\x00\x00\xb4\xb4\x02\x1d\x00\x00\x00\xd3IDAT(\xcf\x85R\xb1\x0e\xc2 \x14\xbcv\xea\xd0\xad\xff\xe1g\xf8W\xe5\x87\x8c\x9b\x0e\xf6\x0f\xbah\xb4k\x1b\xd7\x92h\x87\xa6&D\x0b\xcf\xa1\x94\x02\xa2\xde\xf0\x02\xc7\xbd\x03\x0e""\x02\x00\x05\xc4\xffk\xa4\xd5\xa6\xe1\'b#\xed\xa4\\\xda\xac\xda\xbd\x16^\xab\x1fDYQ\xb1\x86k\x87ykv\xe5YQ\tE\x9a\xa7\t\x92\xf2\xba\xc5\xb6\xcc\xeb\x96\xa4\xc5l\xca\xbcn\x8dfV{\r\xe4\x8c\r\x16o\xdb\x0f\x87\xb3v\x95\xce\xaa\x95\xc9\x8c\xa8\xb8`\x10H\x13Z\xaf>2Q\xce\x9c5\x1c\xbd@\x9a\xa0\x17\xac\xe1\xce\xaa\x9an)\x03\x17\xf5\xcf-\xeds\xd3\x97L\\F{\x0f\xa3\xc2\xfe\xe8%\xa0\x1bv\xa7aTn&D\xb7\xe7H!\xdc\r\x1f\xcc$\xfcg\x14\x10\x9bL\xecj\x1e\xdfc\x807\xf5F\\\xcf\x16\x16"\xa5\x00\x00\x00\x00IEND\xaeB`\x82'))),
    'info_icon': util.img_as_float(io.imread(BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x0f\x00\x00\x00\x0f\x08\x02\x00\x00\x00\xb4\xb4\x02\x1d\x00\x00\x01\xf5IDAT(\xcfURKO\x1aa\x14=\xe7\x9b\x19-\xe3BV"\xaf&\xa2\xc5\xfaJ\x00\x8b\x8b\xb2\xb76m\xff\xa1\xfd\x1f\xb0*\x067\x02ac\x13c\x035\xa9\xbcL-UP\x1e3\xf3\xcd\xedb\xb0\xadwqr\x93{rr\xee\xbd\x87\x93\xc7\x11\x9e\x95\xfc\xedH\x8a\x88\xeb\xb8$\xa9\x14\x15\xcd\xff\x89\x9e\xeb^w\xda7\xfd\x9b\xfb\xbb\xfb\xe5\xf0r$\xb2\x92\x88\'\x0c\xcb\x14\xed\x93$\xc0\xc9x\x14\xc8]]}?==\x15Az3\xbd`-8\xae\xf3\xed\xf2\x92d\xa1PX[K\x81\x00\xc0\xc9\xe3\x10@\xbb\xd3)\x15\x8b\xfb\xfbo\xf6v\xf7\x0c\xcb\x08\x1ci\xcf;\xffz^\xaf\xd7\xdf\x1f\x1d\xc5\x13\t\x01\x14\x00\xc7qO\xca\xe5\xed\xed\x9dl6cZ\xa6;s\x1a\x8d\x863s\x0c\xd3\xccd\xb2;;\xbb_\xca\'\x8e\xe3\x02P [\xad&\xa9\x0e\xf2\x07\x00\x01h\xed\xf7\xbb=\xdf\xd7 @\xe4\xf3yE6[M\x02\x8a`\xaf\xd7\x8f\xc5b\xa6e\x92\x00\x11Z\n}\xf8\xf41d\xdb\x04\x08\x98\x96\x19\x8b\xc7\xfb\xbd>\x02\'\xd3\xe9t\xc9\xb6\x19\xec\x01<\x0cG\x9f\x8f\x8f\x1fF\xff.k\xdb\xf6l6\x03\xa8\x00\xd8!{<\x99<\xbf:\x9f\x90\x04&\xe3\xf1\x8b\xc5\xc5\xb9\xf6j4\xd2\xe9t<O\xcf\xc7Td\x80\xc1\x13t\xb7\xdb]\x8dF\xe7\xec\x8d\xf5W\xbe\xefW\xabU\xa5\x14\t\xc34\xa2\xd1(\x15E\x84\xc4Y\xf5\xcc\xd3\xde\xc6\xfa\x06\x08\xce\xa6\x8f\x10\xb4\xaf\x7f\x14K\xa5\\.\xb7\xf5zK\x19\x86\xd6> \x86a\\\\\\\xd4j\xb5\xc3w\x87/\x93I\x018\x1d\x8f\xb4\xd6Z\xfb\xcdV\xb3R\xa9@\x90No\x86\xec\x90\xd6\xba\xd5j*\xf2m\xa1\x90ZK\x05\x11\xe2\xef\xdb\x9f\x02!H\xd2q\x9dn\xaf7\x18\x0c\x86\xc3Q8\xbc\x1c\x89\xac$\xe3\tk\xc1\x92y\xd2\x84w\xbfn\x95RAf\x04""\x86e*\xa5\x08(E\x02\xbe\xc8S.\xe5\x0f\x85\x0c\xe4\xd9\xc1\xce\xab\xec\x00\x00\x00\x00IEND\xaeB`\x82'))),
    'info_x': util.img_as_float(io.imread(BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x0f\x00\x00\x00\x0f\x08\x02\x00\x00\x00\xb4\xb4\x02\x1d\x00\x00\x01#IDAT(\xcf\x8dR\xc1n\xc20\x0c}\x8e\xfa\x0f\xdb\xce\xa3);\xe4\x9a\x9c\xca\x97S\t\x9a\x03\x13\x1f\x80\xb4\xac\x91\x1a\x86\xc6y\xda\xd0\x148x\x87\xb6\x01\xca&\xcd\x87\xc4v\x9ee\xfb\xe5\xd1\xe1\xeb\x03\x97\xc6\x8c\x1bc\x06\x11\x98\x91QJ\xf57\xdd\xa2\x89\x18 \x80E\x82\xd2\x9f`N\xa7\xe8\xde\th\xdb\xb0\raT\xd0\x866\x84m\n\xc5\x00\xa6\x18\xe3\xbc\xaa\xbc\xf7\xa9\xc0{_\xcd\xab\x18cBg\xc9+\n\t\xf0r\xb18\x1dO\xd3\xa7\xe9f\xb3Y=\xaf\xca\xb2\x94Rv\x93\x12qv\xb9LQ\x14\x00\xac\xb5!\x84\xdd\xee\xad\x9c\xcdd./7\xc8\xfa\xbe\x9c:\x14\xfb\xf7}\xe3\x9b\xc9\xe3DJ9\xac\xd7\xaf"\xae9 \xe7\\\xe3\x9b\xfb\xbb\x87\xc67\xee\xe5e\xc4N\x86\x8ex\x02\x18\xce\xb9\xba\xb6J)c\xccz\xbd\xb6\xd6\x92\x10y\x9e\'\x1a\xcfs\xbbWW\xdbZ)e\x8c\x06\xa0\xb5f\xa0^.\x01\xe4\x93|\xccI\x8cQ)e\xb4I\x7fa\xb4\x06p\x8c\xc7\xf3\xa0\xdf\x87\xcf+q\xf0\xb5\x0e:oH\x8a^\x08\xf8\x97\x89\xb10~\x95\xcb\x10\xfd\x00ln\x80t\xd8\xea\xbe\xc6\x00\x00\x00\x00IEND\xaeB`\x82')))
}


def get_lines_to_remove(orig, goal):
    w, h = orig
    is_r = w / h
    should_r = goal
    # maybe solve the equation instead of bruteforcing it
    if is_r > should_r:     # too wide
        mode = vert
        lines = 0
        while True:
            lines += 1
            if (w - lines) / h <= should_r:
                break
    elif should_r > is_r:   # too high
        mode = hori
        lines = 0
        while True:
            lines += 1
            if w / (h - lines) >= should_r:
                break
    else:
        mode = None
        lines = 0
    return (mode, lines)


def get_resized_image(file, ratio):
    img = util.img_as_float(io.imread(file))
    if len(img.shape) >= 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)

    eimg = filters.sobel(color.rgb2gray(img))
    width = img.shape[1]
    height = img.shape[0]

    mode, rm_paths = get_lines_to_remove((width, height), ratio)
    if mode:
        logger.debug("Carving %s %s paths ", rm_paths, mode)
        outh = transform.seam_carve(img, eimg, mode, rm_paths)
        return outh
    else:
        return img


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def binaryProximitySearch(data, val):
    highIndex = len(data) - 1
    lowIndex = 0
    while highIndex > lowIndex:
            index = (highIndex + lowIndex) // 2
            sub = data[index][0]
            if data[lowIndex][0] == val:
                    return [lowIndex, lowIndex]
            elif sub == val:
                    return [index, index]
            elif data[highIndex][0] == val:
                    return [highIndex, highIndex]
            elif sub > val:
                    if highIndex == index:
                            return sorted([highIndex, lowIndex])
                    highIndex = index
            else:
                    if lowIndex == index:
                            return sorted([highIndex, lowIndex])
                    lowIndex = index
    return sorted([highIndex, lowIndex])


def get_candidate_ad(ads, value):
    center = binaryProximitySearch(ads, value)[0]
    idx = proximity_roll(center, 0, len(ads) - 1, CANDIDATE_NEIGHBORS)
    ratio, path, uses = ads[idx]
    uses += 1
    if uses >= REUSES:
        del ads[idx]
    else:
        ads[idx] = (ratio, path, uses)
    return path


def proximity_roll(center, mini, maxi, scatter):
    lower = max(mini, center - scatter)
    upper = min(maxi, center + scatter)
    return random.randint(lower, upper)


def paste(bg, fg, coord):
    x, y = coord
    fg_w = fg.shape[1]
    fg_h = fg.shape[0]
    bg[y:y + fg_h, x:x + fg_w] = fg[:, :]


def write_box(box, size):
    xmin = box['left']
    xmax = box['right']
    ymin = box['top']
    ymax = box['bottom']
    box = (float(xmin), float(xmax), float(ymin), float(ymax))
    # create annotation
    c = convert(size, box)
    annotation_line = '{} {} {} {} {}\n'.format(0, c[0], c[1], c[2], c[3])
    return annotation_line


def write_boxes(boxes, size):
    annotation_lines = ''
    for c in boxes:
        b = boxes[c]
        annotation_line = write_box(b, size)
        annotation_lines += annotation_line
    return annotation_lines


def generate(im, bb, ads, num=1, mode='bb'):
    logger.info("Generate from %s", im)
    abs_path = os.path.abspath(im)
    rem, _ = os.path.split(abs_path)
    rem, p2 = os.path.split(rem)
    rem, p1 = os.path.split(rem)
    out = "{}_{}".format(p1, p2)
    images = [util.img_as_float(io.imread(im)) for i in range(0, num)]
    # remove alpha channel
    if len(images[0].shape) >= 3 and images[0].shape[2] == 4:
        logger.debug("Removed alpha from image")
        images = [color.rgba2rgb(img) for img in images]
    with open(bb) as bbf:
        boxes = json.load(bbf)
    if mode == "mixed":
        generate_mixed(images, boxes, ads, out)
        return
    if mode == "inter":
        generate_interstitial(images, ads, out)
        return
    if mode in ["bb", "bb+logo"]:
        generate_bb(images, boxes, ads, out, mode)
        return
    if mode == "copy":
        generate_copy(images, boxes, out)
        return
    if mode == "bg":
        generate_background(images, ads, boxes, out)
        return


def generate_copy(images, boxes, out):
    global num_adfree
    if num_adfree < MAX_NUM_ADFREE:
        num_adfree += len(images)
    else:
        logger.info("Created enough ADFREE")
        return
    if boxes:
        logger.error("%s has boxes that should be replaced ... skipped", out)
        return
    logger.info("Copying %s", out)
    tmp_out = '{}-copy'.format(out)
    annotation_path = os.path.join(ANNOTATION_PATH, tmp_out + '.txt')
    image_path = os.path.join(IMAGES_PATH, tmp_out + '.png')
    with open(annotation_path, 'w+') as f:
        f.write('')
    try:
        io.imsave(image_path, images[0])
        logger.info("Saved %s", tmp_out)
    except ValueError:
        logger.error("Failed to save %s", image_path)


def generate_mixed(images, boxes, ads, out):
    try:
        if images[1].shape[1] < WIDTH_SHOULD:
            generate_background(images, ads, boxes, out)
        else:
            if not boxes:
                # no ads, so interstitial or no ads at all
                if random.randint(0, 2) < 2:
                    generate_copy(images, boxes, out)
                else:
                    generate_interstitial(images, ads, out)
            else:
                mode = random.choice(["bb", "bb+logo"])
                generate_bb(images, boxes, ads, out, mode)
    except BaseException as e:
        logging.exception("oops")


def generate_background(images, ads, boxes, out):
    global num_background
    if num_background < MAX_NUM_BACKGROUND:
        num_background += len(images)
    else:
        logger.info("Created enough BACKGROUND")
        return
    logger.info("Adding background to %s", out)
    w = images[0].shape[1]
    h = images[0].shape[0]
    w_diff = WIDTH_SHOULD - w
    h_diff = HEIGHT_SHOULD - h
    right_add = w_diff // 2
    left_add = w_diff - right_add   # left side may be one pixel wider
    left_ratio = left_add / h
    right_ratio = right_add / h
    mode = random.choice(['bb', 'bb+logo'])
    if w_diff < 200:
        logger.error("Size diff too small: %s ... skipped", w_diff)
        return
    if h_diff != 0:
        logger.error("Height does not match, diff: %s ... skipped", h_diff)
        return
    for c in boxes:
        b = boxes[c]
        b['left'] += left_add
        b['right'] += left_add

    for i in zip(images, range(len(images))):
        left_ad_path = get_candidate_ad(ads, left_ratio)    # ratio for left and right is (almost) identical
        right_ad_path = get_candidate_ad(ads, right_ratio)    # ratio for left and right is (almost) identical
        image = i[0]
        left_ad = get_resized_image(left_ad_path, left_ratio)
        left_ad = transform.resize(left_ad, (h, left_add),
                                   anti_aliasing=True, mode="reflect")
        np.clip(left_ad, 0., 1., left_ad)
        right_ad = get_resized_image(right_ad_path, right_ratio)
        right_ad = transform.resize(right_ad, (h, right_add),
                                     anti_aliasing=True, mode="reflect")
        np.clip(right_ad, 0., 1., right_ad)
        # remove scrollbar, add left and right ad + scrollbar
        scrollbar = image[:, -12:]
        image = image[:, :-12]
        image = np.append(left_ad, image, axis=1)
        image = np.append(image, right_ad, axis=1)
        image = np.append(image, scrollbar, axis=1)

        for box in boxes:
            # get info
            box = boxes[box]
            xmin = box['left']
            xmax = box['right']
            ymin = box['top']
            ymax = box['bottom']
            box_width = xmax + 1 - xmin
            box_height = ymax + 1 - ymin
            ratio = box_width / box_height

            ad_path = get_candidate_ad(ads, ratio)
            logger.debug("Chose ad %s", ad_path)
            ad = get_resized_image(ad_path, ratio)
            logger.debug("Resize from %s to %s", ad.shape, (box_height, box_width))
            ad = transform.resize(ad, (box_height, box_width),
                anti_aliasing=True, mode="reflect")
            np.clip(ad, 0., 1., ad)
            # place in image
            paste(image, ad, (xmin, ymin))
            if mode == "bb+logo":
                logo_t = random.randint(0, 4)
                if logo_t == 0:
                    # adchoice only
                    logo = MISC['ac_icon']
                    coord = (xmax - logo.shape[1], ymin + 1)
                    paste(image, logo, coord)
                elif logo_t == 1:
                    # adchoice + close
                    logo = MISC['ac_x']
                    coord = (xmax - logo.shape[1], ymin + 1)
                    paste(image, logo, coord)
                    logo = MISC['ac_icon']
                    coord = (coord[0] - logo.shape[1] - 1, coord[1])
                    paste(image, logo, coord)
                elif logo_t == 2:
                    # info
                    logo = MISC['info_icon']
                    coord = (xmax - logo.shape[1] - 5, ymin + 6)
                    paste(image, logo, coord)
                elif logo_t == 3:
                    # info + close
                    logo = MISC['info_x']
                    coord = (xmax - logo.shape[1] - 5, ymin + 6)
                    paste(image, logo, coord)
                    logo = MISC['info_icon']
                    coord = (coord[0] - logo.shape[1] - 6, coord[1])
                    paste(image, logo, coord)

        tmp_out = '{}-{}-background'.format(out, i[1])
        logger.info("Writing %s", tmp_out)
        label_path = os.path.join(ANNOTATION_PATH, tmp_out + '.txt')
        with open(label_path, 'w') as f:
            annotation_lines = ''
            c = convert((w, h), (0., float(left_add - 1), 0., float(h - 1)))
            annotation_lines += '{} {} {} {} {}\n'.format(0, c[0], c[1], c[2], c[3])
            r_xmin = left_add + w - 12 # subtract scrollbar from width
            r_xmax = r_xmin + right_add - 1
            c = convert((w, h), (float(r_xmin), float(r_xmax), 0., float(h - 1)))
            annotation_lines += '{} {} {} {} {}\n'.format(0, c[0], c[1], c[2], c[3])
            annotation_lines += write_boxes(boxes, (w, h))
            f.write(annotation_lines)

        image_path = os.path.join(IMAGES_PATH, tmp_out + '.png')
        try:
            io.imsave(image_path, image)
            logger.info("Saved %s", tmp_out)
        except ValueError:
            logger.error()


def generate_interstitial(images, ads, out):
    global num_interstitial
    if num_interstitial < MAX_NUM_INTERSTITIAL:
        num_interstitial += len(images)
    else:
        logger.info("Created enough INTERSTISTIAL")
        return
    logger.info("Adding interstitial to %s", out)
    w = images[0].shape[1]
    h = images[0].shape[0]
    for i in zip(images, range(len(images))):
        image = i[0]
        # pick best fitting ad, no need to choose random candidate since ratio is randomized already
        ratio = random.uniform(0.7, 1.5)
        ad_height = random.randint(400, 800)
        ad_width = int(ad_height * ratio)
        ad_path = get_candidate_ad(ads, ratio)
        logger.debug("Chose ad %s", ad_path)

        ad = get_resized_image(ad_path, ratio)
        ad = transform.resize(ad, (ad_height, ad_width),
                anti_aliasing=True, mode="reflect")
        np.clip(ad, 0., 1., ad)
        darkness = 1 - random.uniform(0.5, 0.9)
        image *= darkness
        ad_h = ad.shape[0]
        ad_w = ad.shape[1]
        xmin = w // 2 - ad_w // 2
        xmax = w // 2 + ad_w // 2
        ymin = h // 2 - ad_h // 2
        ymax = h // 2 + ad_h // 2
        paste(image, ad, (xmin, ymin))

        c = convert((w, h), (float(xmin), float(xmax), float(ymin), float(ymax)))
        tmp_out = '{}-{}-interstitial'.format(out, i[1])
        annotation_lines = '{} {} {} {} {}\n'.format(0, c[0], c[1], c[2], c[3])
        annotation_path = os.path.join(ANNOTATION_PATH, tmp_out + '.txt')
        image_path = os.path.join(IMAGES_PATH, tmp_out + '.png')
        with open(annotation_path, 'w+') as f:
            f.write(annotation_lines)
        try:
            io.imsave(image_path, image)
            logger.info("Saved %s", tmp_out)
        except ValueError:
            logger.error("Failed to save %s", i[1])


def generate_bb(images, boxes, ads, out, mode='bb'):
    global num_ad
    global num_adchoice
    if mode == 'bb':
        if num_ad < MAX_NUM_AD:
            num_ad += len(images)
        else:
            if num_adchoice < MAX_NUM_ADCHOICE:
                num_adchoice += len(images)
                mode = 'bb+logo'
            else:
                logger.info("Created enough ADCHOICE")
                return
    else:
        if num_adchoice < MAX_NUM_ADCHOICE:
            num_adchoice += len(images)
        else:
            if num_ad < MAX_NUM_AD:
                num_ad += len(images)
                mode = 'bb'
            else:
                logger.info("Created enough AD")
                return
    logger.info("Filling bounding boxes of %s", out)
    w = images[0].shape[1]
    h = images[0].shape[0]
    annotation_lines = ''
    for box in boxes:
        # get info
        box = boxes[box]
        xmin = box['left']
        xmax = box['right']
        ymin = box['top']
        ymax = box['bottom']
        box = (float(xmin), float(xmax), float(ymin), float(ymax))
        # create annotation
        c = convert((w, h), box)
        annotation_line = '{} {} {} {} {}\n'.format(0, c[0], c[1], c[2], c[3])
        annotation_lines += annotation_line
        # get candidate ads
        box_width = xmax + 1 - xmin
        box_height = ymax + 1 - ymin
        if box_width < 20 and box_height < 20:
            logger.error("Box was too small ... skipped")
            return
        ratio = box_width / box_height
        # upper = ratio + ratio * tolerance
        # lower = ratio + ratio * tolerance
        for img in images:
            # choose ad and rescale it
            ad_path = get_candidate_ad(ads, ratio)
            logger.debug("Chose ad %s", ad_path)
            ad = get_resized_image(ad_path, ratio)
            logger.debug("Resize from %s to %s", ad.shape, (box_height, box_width))
            ad = transform.resize(ad, (box_height, box_width),
                anti_aliasing=True, mode="reflect")
            np.clip(ad, 0., 1., ad)
            # place in image
            logger.debug("Pasting")
            paste(img, ad, (xmin, ymin))
            if mode == "bb+logo" and w > 40 and h > 40:
                logo_t = random.randint(0, 4)
                if logo_t == 0:
                    # adchoice only
                    logo = MISC['ac_icon']
                    coord = (xmax - logo.shape[1], ymin + 1)
                    paste(img, logo, coord)
                elif logo_t == 1:
                    # adchoice + close
                    logo = MISC['ac_x']
                    coord = (xmax - logo.shape[1], ymin + 1)
                    paste(img, logo, coord)
                    logo = MISC['ac_icon']
                    coord = (coord[0] - logo.shape[1] - 1, coord[1])
                    paste(img, logo, coord)
                elif logo_t == 2:
                    # info
                    logo = MISC['info_icon']
                    coord = (xmax - logo.shape[1] - 5, ymin + 6)
                    paste(img, logo, coord)
                elif logo_t == 3:
                    # info + close
                    logo = MISC['info_x']
                    coord = (xmax - logo.shape[1] - 5, ymin + 6)
                    paste(img, logo, coord)
                    logo = MISC['info_icon']
                    coord = (coord[0] - logo.shape[1] - 6, coord[1])
                    paste(img, logo, coord)

    # write to files
    for i in range(0, len(images)):
        if mode == "bb":
            tmp_out = '{}-{}-ads'.format(out, i)
        else:
            tmp_out = '{}-{}-adchoice'.format(out, i)
        logger.info("Writing %s", tmp_out)
        annotation_path = os.path.join(ANNOTATION_PATH, tmp_out + '.txt')
        image_path = os.path.join(IMAGES_PATH, tmp_out + '.png')
        with open(annotation_path, 'w+') as f:
            f.write(annotation_lines)
        try:
            io.imsave(image_path, images[i])
            logger.info("Saved %s", tmp_out)
        except ValueError:
            logger.error("Failed to save %s", i[1])


def analyze_ad(path):
    p = os.path.abspath(path)
    img = io.imread(p)
    ratio = img.shape[1] / img.shape[0]
    # ratio, path, uses
    return (ratio, p, 0)


def ad_list(apath, recreate=False):
    saved = os.path.join(apath, 'saved.txt')
    # load from file
    if os.path.exists(saved):
        if recreate:
            os.remove(saved)
        else:
            with open(saved) as f:
                images = json.load(f)
                logger.info("Loaded %s ads", len(images))
                return images

    # recreate
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS*2) as executor:
        files = [os.path.join(apath, f) for f in os.listdir(apath) if os.path.isfile(os.path.join(apath, f))]
        images = executor.map(analyze_ad, files, chunksize=WORKERS*2)
    images = sorted(images)
    with open(saved, 'w+') as f:
        json.dump(images, f, indent=2)
    logger.info("Loaded %s ads", len(images))
    return images


def main(args):
    images = ad_list(args.ads, args.recreate)
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    if not os.path.exists(ANNOTATION_PATH):
        os.makedirs(ANNOTATION_PATH)
    with open(args.data) as f:
        lines = f.readlines()
        random.shuffle(lines)
        mode = args.mode
        for l in lines:
            d = l.strip()
            im_path = os.path.join(d, SCREENSHOT_NAME)
            bb_path = os.path.join(d, BOX_NAME)
            # generate(im_path, bb_path, images, num=args.num, mode=mode)
            generate(im_path, bb_path, images, num=args.num, mode=mode)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with ads')
    parser.add_argument('data', help='File with image paths')
    parser.add_argument('ads', help='Path to directory with ad images')
    parser.add_argument('-n', '--num', default=1, type=int, help='images to generate')
    # bb for simple replacing bounding boxes, bb+logo for replacing and added logo, inter for interstitials on screenshots without boxes, copy just copies and creates label file, mixed randoms type
    parser.add_argument('--mode', '-m', default='mixed', choices=['bb', 'bb+logo', 'inter', 'bg', 'copy', 'mixed'], help='Type of images to produce')
    parser.add_argument('--recreate', default=False, action='store_true', help='Recompute ads info')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
