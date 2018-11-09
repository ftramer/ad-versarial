import numpy as np
import cv2
import tensorflow as tf
import glob
from utils import to_alpha, get_image_paths
import os
from timeit import default_timer as timer


gray_filter = [299./1000, 587./1000, 114./1000]


def compare_hash(h1, h2):
    h1_flat = h1.reshape(-1)
    h2_flat = h2.reshape(-1)
    if h1_flat.size != h2_flat.size:
        raise TypeError('hashes must be of the same shape.',
                        h1_flat.shape, h2_flat.shape)
    return np.count_nonzero(h1_flat == h2_flat) / (1.0 * len(h1_flat))


def remove_alpha(img_ph):
    alpha = img_ph[:, :, 3:]
    rgb = img_ph[:, :, :3]
    return (1 - alpha/255.) * 255. + alpha/255. * rgb


def phash_dct_tf(img_ph, hash_size=8, highfreq_factor=4, approx=False):
    img_size = hash_size * highfreq_factor
    img_ph = remove_alpha(img_ph)

    img_gray = tf.reduce_sum(img_ph * tf.constant(gray_filter), axis=-1)
    if not approx:
        img_gray = tf.floor(img_gray)
    else:
        img_gray = img_gray - 0.5

    img_gray = tf.expand_dims(img_gray, 0)
    img_gray = tf.expand_dims(img_gray, -1)
    img_resized = tf.image.resize_bicubic(img_gray, (img_size, img_size),
                                          align_corners=True)

    dct = tf.spectral.dct(img_resized[0, :, :, 0])
    dctlowfreq = dct[:hash_size, 1:hash_size + 1]

    return dctlowfreq


def phash_avg_tf(img_ph, h=25, w=25, approx=False):

    img_4d = tf.expand_dims(img_ph, 0)
    img_resized = tf.image.resize_images(img_4d, (h, w),
                                         method=tf.image.ResizeMethod.BICUBIC)

    img_resized = remove_alpha(tf.squeeze(img_resized))
    img_gray = tf.reduce_sum(img_resized * tf.constant(gray_filter), axis=-1)

    if not approx:
        img_gray = tf.floor(img_gray)
    else:
        img_gray = img_gray - 0.5

    img_gray = tf.transpose(img_gray)
    return tf.reshape(img_gray, [-1])


class PHashModel(object):

    def __init__(self, path_to_templates, hash_type, match_threshold=0.8):
        self.match_threshold = match_threshold

        self.img_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))

        if hash_type == "avg":
            self.hash_tensor = phash_avg_tf(self.img_ph)
        else:
            self.hash_tensor = phash_dct_tf(self.img_ph)

        self.sess = tf.Session()

        templates, phashes = self.load_templates(path_to_templates)
        self.templates = templates
        self.phashes = phashes

    def load_templates(self, path_to_templates):
        print("LOADING TEMPLATES...")
        files = glob.glob(path_to_templates + '/*.png')

        templates = []
        phashes = []

        for file in files:
            templates.append(file)
            img = to_alpha(cv2.imread(file, -1))

            phash = self.sess.run(self.hash_tensor,
                                  feed_dict={self.img_ph: img})
            phash = np.sign(phash - np.mean(phash))
            phashes.append(phash)

        print("LOADED {} TEMPLATES".format(len(templates)))
        return templates, phashes

    def match(self, img, verbose=False, img_name=None):
        img = to_alpha(img)
        phash = self.sess.run(self.hash_tensor,
                              feed_dict={self.img_ph: img})
        phash = np.sign(phash - np.mean(phash))

        high_score = 0.0
        for idx, template_hash in enumerate(self.phashes):
            score = compare_hash(phash, template_hash)

            high_score = max(score, high_score)
            if score >= self.match_threshold:
                if verbose:
                    print("\t{} matched on template {} with {} similarity".format(
                        img_name, self.templates[idx], score))
                return True, score

        if verbose:
            print("no match for {}! Highest score was {}".format(
                img_name, high_score))
        return False, high_score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('template_path', type=str)
    parser.add_argument('glob_path', type=str)
    parser.add_argument('--hash_type', choices=['avg', 'dct'], default='avg')
    args = parser.parse_args()

    threshold = 0.8
    PHash = PHashModel(args.template_path, args.hash_type,
                       match_threshold=threshold)

    ad_logo_paths = get_image_paths(args.glob_path)
    print("found {} files to match".format(len(ad_logo_paths)))

    scores = []

    t1 = timer()
    for ad_logo_path in ad_logo_paths:
        ad_logo = cv2.imread(ad_logo_path, -1)
        assert ad_logo is not None

        ad_logo_name = os.path.basename(ad_logo_path)

        _, score = PHash.match(ad_logo,
                               verbose=True,
                               img_name=ad_logo_name)
        scores.append(score)

    t2 = timer()
    print("evaluated {} images in {} seconds".format(len(ad_logo_paths), t2 - t1))

    scores = np.array(scores)
    ad_logo_paths = np.array(ad_logo_paths)

    print(np.sum(scores >= threshold))
    print(ad_logo_paths[scores >= threshold])
    topk = scores.argsort()[-10:][::-1]
    print(ad_logo_paths[topk])


