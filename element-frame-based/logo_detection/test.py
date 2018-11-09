from .yolo_v3 import backbone
import logo_detection.yolo_utils as yolo_utils
import logo_detection.visual as visual
from logo_detection import CKPT_PATH, NAMES_PATH
import cv2
import numpy as np
import tensorflow as tf

input_path = '../data/web/www.cnn.com/frame_6.png'
output_path = None

img = cv2.imread(input_path)
img_resized = cv2.resize(img, (416, 416))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_float = np.expand_dims(img_rgb/255., axis=0)


inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
predictions = backbone(inputs, 1, is_training=False, scope='yolov3')
predictions = yolo_utils.detections_boxes(predictions)
# load weight
saver = yolo_utils.restore_saver()
names = yolo_utils.load_coco_name(NAMES_PATH)

with tf.Session() as sess:
    saver.restore(sess, CKPT_PATH)
    predictions = sess.run(predictions, feed_dict={inputs: img_float})

predictions = yolo_utils.non_max_suppression(predictions)
imgs = [img]

visual.vis(imgs, predictions, (416, 416), names, output_path, convert_rgb=True)
