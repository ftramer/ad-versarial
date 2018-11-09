import time
import cv2

from OCR.tf_tesseract.my_vgsl_model import ctc_loss, ctc_decode
from OCR.ocr_utils import *

from tensorflow.python.ops import state_ops
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import \
    cudnn_rnn_canonical_to_opaque_params, CudnnLSTMSaveable


def init(params, use_gpu=True, skip=1):
    if use_gpu:
        i = 0
        j = 0
        init_ops = []
        units = [64, 96, 96, 512]
        prevs = [16, 64, 96, 96]
        for variable in tf.trainable_variables()[skip:]:
            if 'unknown' in str(variable.get_shape()):
                canonical_w = tf.constant(params[i], dtype=tf.float32)
                canonical_b = tf.constant(params[i + 1], dtype=tf.float32)
                lstm = CudnnLSTMSaveable(
                    num_layers=1,
                    num_units=units[j],
                    input_size=prevs[j],
                    opaque_params=variable
                )
                canonical_w = lstm._tf_to_cudnn_weights(0, canonical_w)
                canonical_b = lstm._tf_to_cudnn_biases(canonical_b)
                opaque_v = cudnn_rnn_canonical_to_opaque_params('lstm',
                                                                1,
                                                                units[j],
                                                                prevs[j],
                                                                canonical_w,
                                                                canonical_b)
                j += 1
                i += 2
                init_op = state_ops.assign(
                    variable,
                    opaque_v,
                    validate_shape=False)

                init_ops.append(init_op)
                continue

            init_op = variable.assign(params[i])
            init_ops.append(init_op)
            i += 1
    else:
        init_ops = []
        for i, variable in enumerate(tf.trainable_variables()[skip:]):
            init_op = variable.assign(params[i])
            init_ops.append(init_op)
    return init_ops


def l2_attack(model, img, target_sl, params, char_map,
              lr=0.01, max_iters=200, const=3., verbose=True,
              use_gpu=True, size_mul=1, target_str=None, target_chars=None,
              fp=False, sim_threshold=4, output_dir="", fname=""):

    h, w, ch = img.shape
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=False)

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        image_var = tf.placeholder(dtype=tf.float32, shape=[h, w, ch])

        height_var = tf.placeholder(dtype=tf.int64, shape=[1])
        width_var = tf.placeholder(dtype=tf.int64, shape=[1])

        target_sparse_label_var = tf.sparse_placeholder(dtype=tf.int64)
        target_sparse_labels = tf.cast(target_sparse_label_var, tf.int32)

        # the variable we're going to optimize over
        modifier = tf.Variable(tf.zeros(shape=(h, w, ch), dtype=tf.float32))
        adv_x = image_var + modifier
        adv_x = tf.clip_by_value(adv_x, 0.0, 255.0)

        round_flag = tf.placeholder(tf.bool)
        val_if_true = tf.cast(tf.cast(adv_x, tf.uint8), tf.float32)
        val_if_false = adv_x
        adv_x = tf.where(round_flag, val_if_true, val_if_false)

        adv_x_preproc = adv_x
        if ch == 4:
            adv_x_preproc = remove_alpha(adv_x)

        adv_x_preprocs = [adv_x_preproc]
        adv_x_preprocs = [preprocess_tf(adv_x_preproc, h, w) 
                          for adv_x_preproc in adv_x_preprocs]
        adv_x_larges = [
            tf.image.resize_images(adv_x_preproc,
                                   (int(size_mul*h), int(size_mul*w)),
                                   method=tf.image.ResizeMethod.BILINEAR)
            for adv_x_preproc in adv_x_preprocs
        ]
        adv_x_larges = [tf.image.rgb_to_grayscale(adv_x_large)
                        for adv_x_large in adv_x_larges]

        model(adv_x_larges[0], height_var, width_var)
        logits = [model(adv_x_large, height_var, width_var, reuse=True)[0]
                  for adv_x_large in adv_x_larges]  
        
        logits_mean = logits[0]
        logits_mask = tf.placeholder(dtype=tf.float32, 
                                     shape=logits_mean.get_shape().as_list())
        text_output = ctc_decode(logits_mean, model.ctc_width)

        if target_chars is not None:
            loss_norm = tf.reduce_sum(tf.square(modifier / 255.0))
            print("target loss using logits_mask with fixed target chars")
            loss_target = -tf.reduce_sum(logits_mask * logits)
            loss = const * loss_target + loss_norm
        else:
            loss_norm = tf.reduce_sum(tf.square(modifier / 255.0))

            if fp:
                print("target loss using CTC")
                loss_target = -ctc_loss(logits_mean, model.ctc_width,
                                        target_sparse_labels)
            else:
                print("target loss using logits_mask")
                loss_target = -tf.reduce_sum(logits_mask * logits)

            if fp:
                loss_target *= -1
            loss = const * loss_target + loss_norm

        init_ops = init(params, use_gpu=use_gpu)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        grads_and_vars = optimizer.compute_gradients(loss, [modifier])
        adv_grad = grads_and_vars[0][0]

        train_ops = optimizer.apply_gradients([(adv_grad, modifier)])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        for init_op in init_ops:
            sess.run(init_op)

        ops = tf.variables_initializer(var_list=[modifier] + new_vars)
        sess.run(ops)

        feed_dict = {
            image_var: img,
            height_var: [int(size_mul * h)],
            width_var: [int(size_mul * w)],
            target_sparse_label_var: target_sl
        }

        i = 0
        adv_img = None
        lower_learning_rate = False
        start_time = time.time()
        while True:
            if i > max_iters:
                break

            curr_logits = \
                sess.run(logits_mean,
                         feed_dict={**{round_flag: False}, **feed_dict})
            top2 = np.argpartition(curr_logits, -2, axis=-1)
            
            decoded1 = decode(top2[:, :, -1], char_map, sparse=False)
            decoded2 = decode(top2[:, :, -2], char_map, sparse=False)

            print(decoded1)

            mask = np.zeros_like(curr_logits)
            if target_chars is not None:
                for j, (char, char_curr) \
                        in enumerate(zip(target_chars, decoded1[0])):
                    if char.lower() != char_curr.lower():
                        mask[0, j, char_map[char.upper()]] = 1.0
                        mask[0, j, char_map[char.lower()]] = 1.0
                        mask[0, j, char_map[char_curr]] = -1.0
                    else:
                        mask[0, j, char_map[char.upper()]] = 0.1
                        mask[0, j, char_map[char.lower()]] = 0.1

                feed_dict[logits_mask] = mask
            else:
                for j, (char1, char2) \
                        in enumerate(zip(decoded1[0], decoded2[0])):
                    if char1.lower() in target_str.lower():
                        mask[0, j, char_map[char1]] = -1.0

                    if char1 != ' ' and not char1.isalnum():
                        mask[0, j, char_map[char1]] = -1.0

                feed_dict[logits_mask] = mask

            _, curr_loss_target, curr_loss_norm, curr_loss = \
                sess.run([train_ops, loss_target, loss_norm, loss],
                         feed_dict={**{round_flag: False}, **feed_dict})

            curr_output, adv_img = \
                sess.run([text_output, adv_x],
                         feed_dict={**{round_flag: True}, **feed_dict})

            curr_output = decode(curr_output, char_map)[0].lower()
            dist = levenshtein(curr_output, target_str.lower())

            print(curr_output, dist)

            if (fp and dist <= sim_threshold) \
                    or (not fp and dist > sim_threshold
                        and target_str.lower() not in curr_output):
                image_to_save = adv_img.astype(np.uint8)
                cv2.imwrite(output_dir + '/{}_adv_{}.png'.format(fname, i),
                            image_to_save)
                return adv_img

            if not lower_learning_rate and \
                ((fp and dist <= sim_threshold+2)
                 or (not fp and dist > sim_threshold-2)):
                lower_learning_rate = True
                lr /= 10.0
                print("new learning rate: {}".format(lr)) 

            if verbose and i % 20 == 0:
                elapsed = time.time() - start_time
                print('\tIteration {}, adv loss={}, target loss={}, '
                      'l2 dist={}, used={:.3f} seconds'.format(
                        i, curr_loss, curr_loss_target, curr_loss_norm, elapsed)
                      )
                
                curr_output = ''.join(
                    [j if ord(j) < 128 else ' ' for j in curr_output])
                print("output: {}".format(curr_output))
                print("dist: {}".format(dist))

                image_to_save = adv_img.astype(np.uint8)
                cv2.imwrite(output_dir + '/{}_adv_{}.png'.format(fname, i),
                            image_to_save)

            i += 1

        return adv_img
