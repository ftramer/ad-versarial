
from OCR.tf_tesseract.my_vgsl_model import MyVGSLImageModel, ctc_decode
from OCR.tf_tesseract.read_params import read_tesseract_params
from OCR.l2_attack import l2_attack, init
from OCR.ocr_utils import *

from tensorflow import app
from tensorflow.python.platform import flags

from utils import *

flags.DEFINE_string('image', "", 'image to load')
flags.DEFINE_integer('target_height', 0, 'If positive, resize image to this height')
flags.DEFINE_integer('iter', 5000, 'Number of iterations for attack')
flags.DEFINE_float('lr', 0.01, 'Learning rate for attack')
flags.DEFINE_float('const', 10.0, 'Balance coefficient in loss function')
flags.DEFINE_integer('use_gpu', -1, 'GPU id (>=0) or cpu (-1)')
flags.DEFINE_string('target', "AdChoices", 'text target')
flags.DEFINE_string('chars_target_file', "", 'file containing specific chars to target')
flags.DEFINE_bool('fp', False, 'set for False Positive generation attack')
flags.DEFINE_bool('start_blank', False, 'set to start from a blank image')
flags.DEFINE_integer('threshold', 4, 'edit distance threshold to target')
flags.DEFINE_string('output_dir', "", 'output directory')

FLAGS = flags.FLAGS

if FLAGS.use_gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(FLAGS.use_gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


def load_example(char_map):
    img_path = FLAGS.image
    
    img = cv2.imread(img_path, -1)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, ch = img.shape
    print(img.shape)

    if FLAGS.fp and FLAGS.start_blank:
        img = np.zeros_like(img)[:, :, :-1]
    
    sl_indice, sl_value, sl_shape = dense_to_sparse(FLAGS.target, char_map)
    sparse_label = tf.SparseTensorValue(sl_indice, sl_value, sl_shape)
    return img, sparse_label


def fine_tune(model, params, char_map, ori_img, adv_img, 
              fp=False, use_gpu=False, threshold=4):

    h, w, ch = ori_img.shape
    assert ori_img.shape == adv_img.shape

    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=False)
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        img_var = tf.placeholder(dtype=tf.float32, 
                                 shape=ori_img.shape)
        height_var = tf.placeholder(dtype=tf.int64, shape=[1], name='height')
        width_var = tf.placeholder(dtype=tf.int64, shape=[1], name='width')
        
        if ch == 4:
            img_var_preproc = remove_alpha(img_var)
        else:
            img_var_preproc = img_var

        img_var_preproc = preprocess_tf(img_var_preproc, h, w)

        size_mul = get_size_mul(h, w, FLAGS.target_height)
        img_large = \
            tf.image.resize_images(img_var_preproc,
                                   (int(size_mul*h),
                                    int(size_mul*w)),
                                   method=tf.image.ResizeMethod.BILINEAR)
        img_large = tf.image.rgb_to_grayscale(img_large)

        logits, _ = model(img_large, height_var, width_var)
        text_output = ctc_decode(logits, model.ctc_width)

        init_ops = init(params, use_gpu=use_gpu, skip=0)
        sess.run(init_ops)

        h_resize, w_resize = new_dims(h, w, FLAGS.target_height)
        logits_np, output = sess.run([logits, text_output],
                                     feed_dict={img_var: adv_img,
                                                height_var: [h_resize],
                                                width_var: [w_resize]})

        s1 = decode(output, char_map)[0]
        print(s1)
        dist = levenshtein(s1.lower(), FLAGS.target.lower())
        print("original: {}".format(dist))
        
        if fp:
            assert dist <= threshold
        else:
            assert dist > threshold

        changed = 0
        pixel_idx = 0
        trials = 0
        max_trials = 3
        while pixel_idx < h*w:
            i = pixel_idx // w
            j = pixel_idx % w

            if np.max(np.abs(adv_img[i, j, :] - ori_img[i, j, :])) < 16.0 \
                or trials >= max_trials:
                
                pixel_idx += 1
                trials = 0
            else:
                old_val = adv_img[i, j, :].copy()

                # trials = 0 => original
                # trials = i => original + i/n * (current - original)
                # trials = n => current
                adv_img[i, j, :] = \
                    ori_img[i, j, :] \
                    + trials / max_trials * (old_val - ori_img[i, j, :])
                
                logits_np, output = \
                    sess.run([logits, text_output],
                             feed_dict={img_var: adv_img,
                                        height_var: [h_resize],
                                        width_var: [w_resize]})

                s1 = decode(output, char_map)[0]
                dist = levenshtein(s1.lower(), FLAGS.target.lower())

                if (not fp and dist <= threshold) \
                        or (fp and dist > threshold):
                    adv_img[i, j, :] = old_val
                    trials += 1
                else:
                    changed += 1
                    pixel_idx += 1
                    trials = 0

        print("reset {} pixels in fine-tuning".format(changed))

        logits_np, output = \
            sess.run([logits, text_output],
                     feed_dict={img_var: adv_img,
                                height_var: [h_resize],
                                width_var: [w_resize]})

        s1 = decode(output, char_map)[0]
        return adv_img, s1


def attack():
    use_gpu = FLAGS.use_gpu >= 0
    char_map = read_all_chars()
    params = read_tesseract_params(use_gpu=use_gpu)
    model = MyVGSLImageModel(use_gpu=use_gpu)
    img, target_sl = load_example(char_map)

    target_chars = None
    if FLAGS.chars_target_file:
        with open(FLAGS.chars_target_file) as f:
            target_chars = f.readline()[1:-1]

    output_dir = FLAGS.output_dir
    safe_mkdir(output_dir)

    fname = os.path.basename(FLAGS.image).split('.')[0]

    h, w, _ = img.shape
    adv_img = l2_attack(model, img, target_sl, params, char_map,
                        max_iters=FLAGS.iter,
                        const=FLAGS.const,
                        lr=FLAGS.lr,
                        use_gpu=use_gpu,
                        fp=FLAGS.fp,
                        size_mul=get_size_mul(h, w, FLAGS.target_height),
                        target_str=FLAGS.target,
                        target_chars=target_chars,
                        sim_threshold=FLAGS.threshold,
                        output_dir=output_dir,
                        fname=fname)

    adv_img = adv_img.astype(np.uint8).astype(np.float32)

    adv_img, adv_text = fine_tune(model, params, char_map, img,
                                  adv_img, fp=FLAGS.fp, use_gpu=use_gpu,
                                  threshold=FLAGS.threshold)
    
    image_to_save = adv_img.astype(np.uint8)
    cv2.imwrite(output_dir + '/{}_fine.png'.format(fname), image_to_save)

    if has_alpha(img):
        img = blend_white(img).astype(np.float32)
    else:
        img = img.astype(np.float32)

    img /= 255.0

    if has_alpha(adv_img):
        adv_img = blend_white(adv_img).astype(np.float32)
    else:
        adv_img = adv_img.astype(np.float32)

    adv_img /= 255.0

    diff = np.linalg.norm((img - adv_img).reshape(-1))

    print(adv_text, diff)
    with open(output_dir + '/{}_norm.txt'.format(fname), 'w') as f:
        f.write("{}\n{:3f}".format(adv_text, diff))


def main(argv):
    del argv
    attack()


if __name__ == '__main__':
    app.run()
