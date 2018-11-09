# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from yolo_v3 import non_max_suppression
from utils import *
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', '', 'Input directory')
tf.app.flags.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', '../models/page_based_yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def normalize(array, min_value=0., max_value=255.):
    """Normalizes the numpy array to (min_value, max_value)
    Args:
        array: The numpy array
        min_value: The min value in normalized array (Default value = 0)
        max_value: The max value in normalized array (Default value = 1)
    Returns:
        The array normalized to range between (min_value, max_value)
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + 1e-8)
    return (max_value - min_value) * normalized + min_value


def filter_inputs(X, Y, w_min, w_max, h_min, h_max, border_px, input_h, input_w, single_ad=True):
    new_X = []
    new_Y = []
    x_h_min = h_max
    x_h_max = h_min
    x_w_min = w_max
    x_w_max = w_min

    for (x, y) in zip(X, Y):
        boxes = y[0]
        good = False
        if len(boxes) == 1 or not single_ad:
            for (box, conf) in boxes:
                x0, y0, x1, y1 = box
                h = y1-y0
                w = x1-x0
                if h_min<=h<=h_max and w_min<=w<=w_max \
                    and y0-border_px > 0 and y1+border_px < input_h:

                    good = True 
                    x_h_min = min(x_h_min, h)
                    x_h_max = max(x_h_max, h)
                    x_w_min = min(x_w_min, w)
                    x_w_max = max(x_w_max, w)

        if good:
            new_X.append(x)
            new_Y.append(y)

    return np.stack(new_X, axis=0), new_Y, [x_h_min, x_h_max, x_w_min, x_w_max]


def main(argv=None):
    np.random.seed(0)

    classes = load_coco_names(FLAGS.class_names)

    input_h = 1013
    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])

    X_train_paths, Y_train = get_input_files_and_labels(FLAGS.input_dir + "/train/", input_h, input_w)
    X_test_paths, Y_test = get_input_files_and_labels(FLAGS.input_dir + "/test/", input_h, input_w)
    print(len(X_train_paths), len(X_test_paths))
    X_test = np.array([load_image(path) for path in X_test_paths])

    epochs = 201
    batch_size = 4

    # set this to get a grad-cam
    vis_gradients = False

    if not vis_gradients:
        mask_tile = 8
        mask_val = np.zeros((input_h//mask_tile, input_w//mask_tile, 3), dtype=np.float32)
        mask = tf.Variable(initial_value=mask_val, dtype=tf.float32)

        slack_h = input_h - mask_val.shape[0]*mask_tile
        slack_w = input_w - mask_val.shape[1]*mask_tile
        tiled_mask = tf.image.pad_to_bounding_box(tf.tile(mask, [mask_tile, mask_tile, 1]), slack_h//2, slack_w//2, input_h, input_w)

        alpha = 0.01
        masked_input = tf.clip_by_value((1-alpha) * inputs + alpha * tiled_mask, 0, 255)
        inputs_resized = tf.image.resize_images(masked_input, (FLAGS.size, FLAGS.size),
                                                align_corners=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if vis_gradients:
        safe_mkdir("output/grads")
        inputs_resized2 = tf.image.resize_images(inputs, (FLAGS.size, FLAGS.size), align_corners=True)
        detections2, boxes_tensor2 = init_yolo(sess, inputs_resized2, len(classes), FLAGS.weights_file, header_size=4)
        loss2 = tf.reduce_sum(boxes_tensor2[:, :, 4])
        grad_img = tf.gradients(loss2, inputs)[0]

        feed_dict = {inputs: X_test}

        curr_grad_img = np.abs(sess.run(grad_img, feed_dict=feed_dict))

        for j in range(len(X_test)):
            g = curr_grad_img[j]

            g = cv2.GaussianBlur(g, (25, 25), 0)

            g = np.sign(g) * np.sqrt(np.abs(g))
            grad_norm = normalize(np.mean(g, axis=-1))

            cam = cv2.applyColorMap(np.uint8(grad_norm), cv2.COLORMAP_JET)
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

            gimg = Image.fromarray(cam.astype(np.uint8))
            gimg.save("output/grads/cam_{}.png".format(j))

            cam = np.float32(cam) + np.float32(X_test[j])
            cam = 255 * cam / np.max(cam)

            gimg = Image.fromarray(cam.astype(np.uint8))
            gimg.save("output/grads/grad_{}.png".format(j))

        return
    else:
        detections, boxes_tensor = init_yolo(sess, inputs_resized, len(classes), FLAGS.weights_file, header_size=4)
        loss = tf.nn.relu(boxes_tensor[:, :, 4] - 0.1 * FLAGS.conf_threshold)
        grad = tf.gradients(tf.reduce_sum(loss), mask)[0]

        full_grad = tf.placeholder(dtype=tf.float32, shape=mask.shape)
        opt = tf.train.AdamOptimizer(1.0)
        assign_op = opt.apply_gradients([(full_grad, mask)])
        sess.run(tf.variables_initializer(opt.variables()))
        assign_eps_op = tf.assign(mask, tf.clip_by_value(mask, 0, 255))

        safe_mkdir("output/overlay")

    for epoch in range(epochs):
        if epoch % 50 == 0:

            mask_np = sess.run(mask)
            Image.fromarray(mask_np.astype(np.uint8)).save("output/overlay/mask_{}.png".format(epoch))

            feed_dict = {inputs: X_test}

            curr_loss, detected_boxes, curr_inputs = batch_eval(sess, [loss, boxes_tensor, masked_input], feed_dict)

            num_detect = []
            num_evaded = 0
            for j in range(len(X_test)):
                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                img = Image.fromarray(curr_inputs[j].astype(np.uint8))
                img.save("output/overlay/img_{}_{}.png".format(epoch, j))

                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
                img.save("output/overlay/img_boxes_{}_{}.png".format(epoch, j))
                img.close()

                ground_truth = 0 if 0 not in Y_test[j] else len(Y_test[j][0])
                if len(filtered_boxes) != 0:
                    if len(filtered_boxes[0]) < ground_truth:
                        num_evaded += 1
                    num_detect.append("{}/{}".format(len(filtered_boxes[0]), ground_truth))
                else:
                    num_evaded += 1
                    num_detect.append("{}/{}".format(0, ground_truth))
            
            print('test loss={:.3f}'.format(np.sum(curr_loss) / len(X_test)),
                  'num_boxes={}'.format(num_detect))
            print("evaded {} ads".format(num_evaded))

        batch_idx = np.random.choice(len(X_train_paths), batch_size, replace=False)
        X_batch = np.array([load_image(X_train_paths[idx]) for idx in batch_idx])
        Y_batch = [Y_train[idx] for idx in batch_idx]

        jitter_x_low, jitter_x_high = -500, 500
        jitter_y_low, jitter_y_high = -50, 50
        jitters_x = np.zeros(len(X_batch))
        jitters_y = np.zeros(len(X_batch))

        for batch_idx in range(len(X_batch)):
            boxes = Y_batch[batch_idx][0]
            for (box, conf) in boxes:
                x0, y0, x1, y1 = box

                h = y1-y0
                w = x1-x0

                low_x = max(-x0, jitter_x_low)
                high_x = min(input_w-x1, jitter_x_high)
                jitter_x = np.random.randint(low_x, high_x)
                low_y = max(-y0, jitter_y_low)
                high_y = min(input_h-y1, jitter_y_high)
                jitter_y = np.random.randint(low_y, high_y)
                jitters_x[batch_idx] = jitter_x
                jitters_y[batch_idx] = jitter_y

                ad = X_batch[batch_idx, y0:y1, x0:x1, :].copy()
                background = X_batch[batch_idx, min(y0+5, input_h-1), min(x1+5, input_w-1), :]
                X_batch[batch_idx, y0-5:y1+5, x0:x1, :] = background
                y0 = y0 + jitter_y
                x0 = x0 + jitter_x
                X_batch[batch_idx, y0:y0+h, x0:x0+w, :] = ad

        max_steps = 40
        i = 0

        num_original = []

        while i < max_steps:
            i += 1

            feed_dict = {inputs: np.clip(X_batch, 0, 255)}

            curr_grad, curr_loss, detected_boxes = \
                sess.run([grad, loss, boxes_tensor], feed_dict=feed_dict)

            num_detect = []

            num_evaded = 0
            num_detected = 0
            for j in range(batch_size):

                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                if i == 1:
                    if len(filtered_boxes) != 0:
                        num_original.append(len(filtered_boxes[0]))
                    else:
                        num_original.append(0)

                if len(filtered_boxes) != 0:
                    if len(filtered_boxes[0]) < num_original[j]:
                        num_evaded += num_original[j] - len(filtered_boxes[0])

                    num_detect.append("{}/{}".format(len(filtered_boxes[0]), num_original[j]))
                    num_detected += 1
                else:
                    num_evaded += num_original[j]
                    num_detect.append("{}/{}".format(0, num_original[j]))

            print(epoch, i, 'loss={:.3f}'.format(np.sum(curr_loss) / batch_size),
                  'num_boxes={}'.format(num_detect))

            if num_detected == 0:
                i = 1000
            else:
                sess.run(assign_op, feed_dict={full_grad: curr_grad / (np.linalg.norm(curr_grad) + 1e-8)})
                sess.run(assign_eps_op)


if __name__ == '__main__':
    tf.app.run()
