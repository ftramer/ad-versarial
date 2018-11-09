import os
import cv2

from OCR.tf_tesseract.my_vgsl_model import MyVGSLImageModel, ctc_decode
from OCR.tf_tesseract.read_params import read_tesseract_params
from OCR.ocr_utils import *
from OCR.l2_attack import init

from utils import to_alpha, get_image_paths

from tensorflow import app
from tensorflow.python.platform import flags

from timeit import default_timer as timer

flags.DEFINE_string('glob_path', "", 'images to load')
flags.DEFINE_integer('target_height', 0, 'Resize image to this height')
flags.DEFINE_string('target', "adchoices", 'text target')
flags.DEFINE_integer('use_gpu', -1, 'GPU id (>=0) or cpu (-1)')

FLAGS = flags.FLAGS

if FLAGS.use_gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(FLAGS.use_gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


class OCRModel(object):

    def __init__(self, target, target_height=0, target_threshold=4):
        self.target_txt = target
        self.target_height = target_height
        self.sim_threshold = target_threshold

        use_gpu = FLAGS.use_gpu >= 0
        self.char_map = read_all_chars()
        params = read_tesseract_params(use_gpu=use_gpu)
        model = MyVGSLImageModel(use_gpu=use_gpu)

        self.img_var = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))
        self.h_orig_var = tf.placeholder(dtype=tf.int64, shape=[1])
        self.w_orig_var = tf.placeholder(dtype=tf.int64, shape=[1])

        self.h_resized_var = tf.placeholder(dtype=tf.int64, shape=[1])
        self.w_resized_var = tf.placeholder(dtype=tf.int64, shape=[1])
        self.resized_dims_var = tf.cast(
            tf.concat([self.h_resized_var, self.w_resized_var], axis=0),
            tf.int32)

        img_preproc = self.img_var
        img_preproc = remove_alpha(img_preproc)
        img_preproc = preprocess_tf(img_preproc,
                                    self.h_orig_var[0],
                                    self.w_orig_var[0])

        img_large = tf.image.resize_images(img_preproc, self.resized_dims_var,
                                           method=tf.image.ResizeMethod.BILINEAR)
        img_large = tf.image.rgb_to_grayscale(img_large)

        logits, _ = model(img_large,  self.h_resized_var, self.w_resized_var)
        self.text_output = ctc_decode(logits, model.ctc_width)

        init_ops = init(params, use_gpu=use_gpu, skip=0)
        self.sess = tf.Session()
        self.sess.run(init_ops)

    def match(self, img, verbose=False, img_name=None):
        img = to_alpha(img)
        h, w, _ = img.shape
        if h < 5 or w < 5:
            if verbose:
                print("Skipping {}, too small".format(img_name))
            return False, len(self.target_txt)

        feed_dict = {
            self.img_var: img,
            self.h_orig_var: [h],
            self.w_orig_var: [w],
            self.h_resized_var: [h],
            self.w_resized_var: [w]
        }

        mul = 1
        if self.target_height and h < self.target_height:
            mul = get_size_mul(h, w, self.target_height)

        feed_dict[self.h_resized_var] = [mul * h]
        feed_dict[self.w_resized_var] = [mul * w]
        output2 = self.sess.run(self.text_output, feed_dict=feed_dict)
        s2 = decode(output2, self.char_map)[0]
        print(s2)
        dist2 = levenshtein(s2.lower(), self.target_txt.lower())
        if self.target_txt.lower() in s2.lower():
            dist2 = 0

        dist = dist2

        if dist <= self.sim_threshold:
            if verbose:
                print("{} matched with distance {}".format(img_name, dist))
            return True, dist
        else:
            if verbose:
                print("no match for {}! Distance {}".format(img_name, dist))
            return False, dist


def main(argv):
    del argv
    threshold = 4
    OCR = OCRModel(FLAGS.target, FLAGS.target_height,
                   target_threshold=threshold)

    ad_logo_paths = get_image_paths(FLAGS.glob_path)
    print("found {} files to match".format(len(ad_logo_paths)))

    scores = []

    t1 = timer()

    for ad_logo_path in ad_logo_paths:
        ad_logo = cv2.imread(ad_logo_path, -1)
        assert ad_logo is not None

        ad_logo_name = os.path.basename(ad_logo_path)

        _, score = OCR.match(ad_logo, verbose=True, img_name=ad_logo_name)
        scores.append(score)

    t2 = timer()

    print("evaluated {} images in {} seconds".format(len(ad_logo_paths), t2-t1))

    scores = np.array(scores)
    ad_logo_paths = np.array(ad_logo_paths)

    print(np.sum(scores <= threshold))
    print(ad_logo_paths[scores <= threshold])
    topk = scores.argsort()[:10]
    print(list(zip(ad_logo_paths[topk], scores[topk])))


if __name__ == '__main__':
    app.run()
