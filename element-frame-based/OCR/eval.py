import os
import cv2

from OCR.tf_tesseract.my_vgsl_model import MyVGSLImageModel, ctc_decode
from OCR.tf_tesseract.read_params import read_tesseract_params
from OCR.ocr_utils import *
from OCR.l2_attack import init

from tensorflow import app
from tensorflow.python.platform import flags

from timeit import default_timer as timer

flags.DEFINE_string('image', "", 'image to load')
flags.DEFINE_integer('target_height', 0, 'Resize image to this height')
flags.DEFINE_string('target', "adchoices", 'text target')
flags.DEFINE_integer('use_gpu', -1, 'GPU id (>=0) or cpu (-1)')
flags.DEFINE_bool('timeit', False, 'time the execution')

FLAGS = flags.FLAGS

if FLAGS.use_gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(FLAGS.use_gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


def eval():
    use_gpu = FLAGS.use_gpu >= 0
    char_map = read_all_chars()
    params = read_tesseract_params(use_gpu=use_gpu)
    model = MyVGSLImageModel(use_gpu=use_gpu)

    img = cv2.imread(FLAGS.image, -1)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, ch = img.shape

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=False)
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        img_var = tf.placeholder(dtype=tf.float32, shape=(None, None, ch))
        height_var = tf.placeholder(dtype=tf.int64, shape=[1], name='height')
        width_var = tf.placeholder(dtype=tf.int64, shape=[1], name='width')
        size_var = tf.placeholder(dtype=tf.int32, shape=[2], name='size')

        img_preproc = img_var
        if ch == 4:
            img_preproc = remove_alpha(img_var)

        size_mul = get_size_mul(h, w, target_height=FLAGS.target_height)
        
        img_preproc = preprocess_tf(img_preproc, height_var[0], width_var[0])

        img_large = tf.image.resize_images(img_preproc, size_mul*size_var, 
                                           method=tf.image.ResizeMethod.BILINEAR)
        img_large = tf.image.rgb_to_grayscale(img_large)

        logits, _ = model(img_large, size_mul*height_var, size_mul*width_var)
        text_output = ctc_decode(logits, model.ctc_width)
        text_output2 = ctc_decode(logits, model.ctc_width, beam=True)

        init_ops = init(params, use_gpu=use_gpu, skip=0)
        sess.run(init_ops)

        if FLAGS.timeit:
            t1 = timer()

            n = 100
            for i in range(n):
                h = np.random.randint(low=40, high=80)
                w = np.random.randint(low=150, high=200)
                img = np.zeros(shape=(h, w, ch), dtype=np.float32)
                sess.run(text_output, feed_dict={img_var: img,
                                                 size_var: [h, w],
                                                 height_var: [h],
                                                 width_var: [w]})
            
            t2 = timer()
            print("time for {} images: {:.3f} s".format(n, t2 - t1))
        else:
            logits_np, output, output2 = sess.run(
                          [logits, text_output, text_output2],
                          feed_dict={img_var: img,
                                     size_var: [h, w],
                                     height_var: [h],
                                     width_var: [w]})

            s1 = decode(output, char_map)[0]
            s2 = decode(output2, char_map)[0]
            labels = np.argmax(logits_np, axis=-1)
            print(decode(labels, char_map, sparse=False))

            dist1 = levenshtein(s1.lower(), FLAGS.target.lower())
            dist2 = levenshtein(s2.lower(), FLAGS.target.lower())

            print(s1, dist1)
            print(s2, dist2)


def main(argv):
    del argv
    eval()


if __name__ == '__main__':
    app.run()
