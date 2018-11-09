# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

PIXEL_VALUE = [50, 90, 140, 180, 220]
np.random.shuffle(PIXEL_VALUE)
COLOR_MAP = tuple(product(PIXEL_VALUE, PIXEL_VALUE, PIXEL_VALUE))[:80]
COLOR_MAP = (255, 0, 0) + COLOR_MAP


def vis(imgs, predictions, input_size, coco_name, output=None, convert_rgb=False):
  """Visualize
  Params:
    imgs: list, consists of PIL.Image object
    inputs_sizes: tuple, (w, h)
  """

  for img, prediction in zip(imgs, predictions):
    for label, bboxes in prediction.items():
      for bbox, confidence in bboxes:
        bbox = descale(img.shape[-2::-1], input_size, bbox)
        print(coco_name[label], ':', confidence)
        print('--- bbox:', bbox)
        cv2.rectangle(img, bbox[:2], bbox[2:4], COLOR_MAP[label], 1)

    if output:
      cv2.imwrite(output, img)

    if convert_rgb:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()


def descale(orignal_size, input_size, bbox):

  scale = np.divide(orignal_size, input_size)
  bbox = np.reshape((np.reshape(bbox,[2,2])*scale),-1)
  print(scale)
  print(bbox)
  return tuple(bbox.astype(np.int))
