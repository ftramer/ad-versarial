# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from yolo3.model import yolo_body, yolo_loss, yolo_eval, preprocess_true_boxes
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input
from PIL import ImageFont
from utils import *

import tensorflow as tf
import numpy as np

src_pos = [(566, 722), (1151, 1012)]

target_pos = [(564, 75), (1148, 571)]

anchors = np.asarray([[10, 13], [16, 30], [33, 23],
                      [30, 61], [62, 45], [59, 119],
                      [116, 90], [156, 198], [373, 326]]).astype(np.float32)


def draw_boxes(img, out_boxes, out_scores, out_classes, input_h=1013, input_w=1919):

    image = img.astype(np.uint8)
    font = ImageFont.truetype(font='keras-yolo3/font/FiraMono-Medium.otf', size=18)

    ratio_h = input_h / 416.0
    ratio_w = input_w / 416.0
    pil_image = Image.fromarray(image.astype(np.uint8))
    for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = "Ad"
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(pil_image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box

            top *= ratio_h
            bottom *= ratio_h
            left *= ratio_w
            right *= ratio_w

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            color = tuple([255, 0, 0])
            cor = (left, top, right, bottom)
            line = (cor[0],cor[1],cor[0],cor[3])
            draw.line(line, fill=color, width=10)
            line = (cor[0],cor[1],cor[2],cor[1])
            draw.line(line, fill=color, width=10)
            line = (cor[0],cor[3],cor[2],cor[3])
            draw.line(line, fill=color, width=10)
            line = (cor[2],cor[1],cor[2],cor[3])
            draw.line(line, fill=color, width=10)

            draw.text(cor[:2], '{} {:.2f}%'.format("ad", score * 100), fill=tuple([0, 0, 255]), font=font)
            del draw

    return pil_image


def main(argv=None):
    np.random.seed(0)

    safe_mkdir("output/abuse")

    input_h = 1013
    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])

    np.random.seed(0)

    img = np.asarray(Image.open('../data/page_based/tj.png')).astype(np.float32)

    inv_ratio_h = 416.0 / input_h
    inv_ratio_w = 416.0 / input_w
    target_box = np.asarray([[[target_pos[0][0] * inv_ratio_w, target_pos[0][1] * inv_ratio_h,
                               target_pos[1][0] * inv_ratio_w, target_pos[1][1] * inv_ratio_h,
                               0]]])
    true_boxes = preprocess_true_boxes(target_box, [416, 416], anchors, 1)

    eps = 4.0
    epochs = 500

    mask_h = src_pos[1][1] - src_pos[0][1]
    mask_w = src_pos[1][0] - src_pos[0][0]
    mask_val = img[src_pos[0][1]:src_pos[1][1], src_pos[0][0]:src_pos[1][0]]

    mask = tf.Variable(initial_value=mask_val, dtype=tf.float32)
    padded_mask = tf.image.pad_to_bounding_box(mask, src_pos[0][1], src_pos[0][0],
                                               tf.shape(inputs)[1], tf.shape(inputs)[2])

    black_box = tf.ones([mask_h, mask_w, 3], dtype=tf.float32)
    black_mask = 1.0 - tf.image.pad_to_bounding_box(black_box, src_pos[0][1], src_pos[0][0],
                                                    tf.shape(inputs)[1], tf.shape(inputs)[2]),

    blacked_inputs = tf.multiply(inputs, black_mask)
    masked_input = tf.clip_by_value(tf.add(blacked_inputs, padded_mask), 0, 255)

    inputs_resized = tf.image.resize_images(masked_input, (416, 416), align_corners=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    model = load_model('../models/page_based_yolov3.h5')
    model.layers.pop(0)
    newInput = Input(tensor=inputs_resized / 255.)
    newOut = model(newInput)
    model = Model(newInput, newOut)

    y_true = [tf.placeholder(shape=(1, 416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], \
        len(anchors)//3, 1+5), dtype=tf.float32) for l in range(3)]

    print([y.get_shape().as_list() for y in y_true])

    loss = yolo_loss([*model.output, *y_true], anchors, 1)
    grad = tf.gradients(loss, mask)[0]
    opt = tf.train.AdamOptimizer(10.0)
    grad_ph = tf.placeholder(shape=grad.get_shape().as_list(), dtype=tf.float32)
    assign_op = opt.apply_gradients([(grad_ph, mask)])
    sess.run(tf.variables_initializer(opt.variables()))

    assign_eps_op = tf.assign(mask, tf.clip_by_value(mask, mask_val-eps, mask_val+eps))

    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(model.output, anchors, 1, input_image_shape, score_threshold=0.5, iou_threshold=0.4)

    time_since_success = np.inf
    for i in range(epochs):

        curr_grad, curr_loss, curr_img, out_boxes, out_scores, out_classes = sess.run(
            [grad, loss, masked_input, boxes, scores, classes],
            feed_dict={
                inputs: np.expand_dims(img, 0).astype(np.float32),
                input_image_shape: [416, 416],
                y_true[0]: np.asarray(true_boxes[0]),
                y_true[1]: np.asarray(true_boxes[1]),
                y_true[2]: np.asarray(true_boxes[2]),
                K.learning_phase(): 0
            })

        num_detect = len(out_boxes)
        print('test loss={:.3f}'.format(curr_loss),
              'num_boxes={}'.format(num_detect))

        sess.run(assign_op, feed_dict={grad_ph: curr_grad / (np.linalg.norm(curr_grad.reshape(-1)) + 1e-8)}) 
        sess.run(assign_eps_op)

        if ((i % 50 == 0) or (time_since_success > 50)) and num_detect > 0:
            img1 = draw_boxes(curr_img[0].astype(np.uint8), out_boxes, out_scores, out_classes)
            img1.save("output/abuse/tj_{}.png".format(i))
            plt.imshow(img1)
            plt.show()
            time_since_success = 0
        else:
            time_since_success += 1


if __name__ == '__main__':
    tf.app.run()
