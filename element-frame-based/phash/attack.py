import tensorflow as tf
from phash.model import phash_avg_tf, phash_dct_tf, compare_hash, PHashModel
from utils import *


if __name__ == '__main__':
    np.random.seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('src_logo', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('template_path', type=str)
    parser.add_argument('--hash_type', choices=['avg', 'dct'], default='avg')
    args = parser.parse_args()

    src_logo_file = args.src_logo
    src_logo = cv2.imread(src_logo_file, -1)
    assert(has_alpha(src_logo))
    src_logo = cv2.cvtColor(src_logo, cv2.COLOR_BGRA2RGBA)
    src_logo = src_logo.copy()

    fname = os.path.basename(src_logo_file).split('.')[0]

    output_dir = args.output_dir
    safe_mkdir(output_dir)

    sess = tf.Session()
    img_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 4))

    src_hash = None

    eps = 8.0
    n = 1000
    alpha = 1.0

    threshold = 0.8
    PHash = PHashModel(args.template_path, args.hash_type,
                       match_threshold=threshold)

    if args.hash_type == "avg":
        hash_tensor = phash_avg_tf(img_ph)
        src_hash = sess.run(hash_tensor, feed_dict={img_ph: src_logo})
        src_mean = np.mean(src_hash)

        #print(src_hash)
        #print(src_mean)
        #print(len(src_hash))
        #print("min score with eps={}: {}".format(eps,
        #    1.0 - (1.0 * len(src_hash[np.abs(src_hash - src_mean) < 2*eps]))
        #                                         / len(src_hash)))

        src_hash = np.sign(src_hash - src_mean)

        for shift in [(0, 0), (1, 0), (2, 0),
                       (1, 1), (2, 1), (2, 2),
                       (3, 0), (3, 1), (3, 2), (3, 3)]:

            shift_x = shift[0]
            shift_y = shift[1]
            img_adv = np.zeros((src_logo.shape[0] + shift_y,
                                src_logo.shape[1] + shift_x,
                                4))
            if shift_y > 0:
                img_adv[:-shift_y, shift_x:, :] = src_logo
            else:
                img_adv[:, shift_x:, :] = src_logo

            img_adv = img_adv.astype(np.uint8)
            adv_hash = sess.run(hash_tensor, feed_dict={img_ph: img_adv})
            adv_hash = np.sign(adv_hash - np.mean(adv_hash))

            _, score = PHash.match(img_adv,
                                   verbose=True,
                                   img_name=fname)
            if score < 0.8:
                with open(output_dir + '/{}.txt'.format(fname), 'w') as f:
                    f.write("{} {}".format(shift_x, shift_y))
                break

    else:
        hash_tensor = phash_dct_tf(img_ph)
        src_hash = sess.run(hash_tensor, feed_dict={img_ph: src_logo})
        src_hash = np.sign(src_hash - np.mean(src_hash))

        tensor_adv = phash_dct_tf(img_ph, approx=True)
        soft_signs = tf.nn.softsign(
            tensor_adv - tf.reduce_mean(tensor_adv))

        loss = tf.reduce_sum(tf.square(soft_signs - src_hash))
        grad = tf.gradients(loss, img_ph)[0]

        img_adv = src_logo.copy().astype(np.float32)

        for i in range(n):
            curr_hash = sess.run(hash_tensor, feed_dict={img_ph: img_adv})
            curr_avg = np.mean(curr_hash)
            curr_hash = np.sign(curr_hash - curr_avg)

            grad_np, curr_loss = sess.run([grad, loss],
                                          feed_dict={img_ph: img_adv})
            print(i, curr_loss, compare_hash(curr_hash, src_hash))

            img_adv += alpha * np.sign(grad_np)
            img_adv = np.clip(img_adv, 0, 255)
            img_adv = np.clip(img_adv, src_logo - eps, src_logo + eps)

    img_adv = img_adv.astype(np.uint8)
    adv_hash = sess.run(hash_tensor, feed_dict={img_ph: img_adv})
    adv_hash = np.sign(adv_hash - np.mean(adv_hash))

    print("adv hash: {}".format(compare_hash(src_hash, adv_hash)))

    cv2.imwrite(output_dir + '/{}_{}_adv.png'.format(fname, args.hash_type),
                cv2.cvtColor(img_adv, cv2.COLOR_RGBA2BGRA))

