import tensorflow as tf
from phash.model import phash_avg_tf, phash_dct_tf, compare_hash
from utils import *


TYPE = "avg"

if __name__ == '__main__':
    np.random.seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target_logo', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--hash_type', choices=['avg', 'dct'], default='avg')
    args = parser.parse_args()

    target_logo_file = args.target_logo
    target_logo = cv2.imread(target_logo_file, -1)
    target_logo = cv2.cvtColor(target_logo, cv2.COLOR_BGRA2RGBA)
    target_logo_orig = target_logo.copy()

    output_dir = args.output_dir
    safe_mkdir(output_dir)

    sess = tf.Session()
    img_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))

    hash_tensor, grad, loss = None, None, None

    if args.hash_type == "avg":
        hash_tensor = phash_avg_tf(img_ph)
    else:
        hash_tensor = phash_dct_tf(img_ph)

    target_hash = sess.run(hash_tensor, feed_dict={img_ph: target_logo})
    target_hash = np.sign(target_hash - np.mean(target_hash))

    tensor_adv = phash_avg_tf(img_ph, approx=True)
    soft_signs = tf.nn.softsign(
        tensor_adv - tf.reduce_mean(tensor_adv))

    loss = tf.reduce_sum(tf.square(soft_signs - target_hash))
    grad = tf.gradients(loss, img_ph)[0]

    eps = 2.0
    n = 50
    alpha = 1.0

    img_adv = 255 * np.ones_like(target_logo).astype(np.float32)
    img_adv[:, :, 3:] = 5
    img_adv_src = img_adv.copy()

    for i in range(n):
        curr_hash = sess.run(hash_tensor, feed_dict={img_ph: img_adv})
        curr_avg = np.mean(curr_hash)
        curr_hash = np.sign(curr_hash - curr_avg)
        grad_np, curr_loss = sess.run([grad, loss], feed_dict={img_ph: img_adv})
        print(i, curr_loss, compare_hash(curr_hash, target_hash))

        img_adv -= alpha * np.sign(grad_np)
        img_adv = np.clip(img_adv, 0, 255)
        img_adv = np.clip(img_adv, img_adv_src - eps, img_adv_src + eps)

    img_adv = img_adv.astype(np.uint8)
    adv_hash = sess.run(hash_tensor, feed_dict={img_ph: img_adv})
    adv_hash = np.sign(adv_hash - np.mean(adv_hash))

    print("junk hash: {}".format(compare_hash(target_hash, adv_hash)))

    cv2.imwrite(output_dir + '/phash_{}_junk_orig.png'.format(TYPE),
                cv2.cvtColor(target_logo, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(output_dir + '/phash_{}_junk_adv.png'.format(TYPE),
                cv2.cvtColor(img_adv, cv2.COLOR_RGBA2BGRA))
