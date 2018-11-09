import numpy as np
import tensorflow as tf
import cv2
from phash.model import phash_avg_tf, phash_dct_tf, compare_hash

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('src_logo', type=str)
    parser.add_argument('target_logo', type=str)
    parser.add_argument('--hash_type', choices=['avg', 'dct'], default='avg')
    args = parser.parse_args()

    np.random.seed(0)
    src_logo_file = args.src_logo
    src_logo = cv2.imread(src_logo_file, -1)
    src_logo_orig = src_logo.copy()

    target_logo_file = args.target_logo
    target_logo = cv2.imread(target_logo_file, -1)
    target_logo_orig = target_logo.copy()

    hash_type = args.hash_type

    sess = tf.Session()
    img_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))

    hash_tensor, grad, loss = None, None, None

    if hash_type == "avg":
        hash_tensor = phash_avg_tf(img_ph)
    else:
        hash_tensor = phash_dct_tf(img_ph)

    src_hash = sess.run(hash_tensor, feed_dict={img_ph: src_logo})
    src_hash = np.sign(src_hash - np.mean(src_hash))

    target_hash = sess.run(hash_tensor, feed_dict={img_ph: target_logo})
    target_hash = np.sign(target_hash - np.mean(target_hash))

    white = 255 * np.ones_like(src_logo)
    white_hash = sess.run(hash_tensor, feed_dict={img_ph: white})
    white_hash = np.sign(white_hash - (np.mean(white_hash) - 0.001))

    print("src vs target: {}".format(compare_hash(src_hash, target_hash)))
    print("src vs white: {}".format(compare_hash(src_hash, white_hash)))
