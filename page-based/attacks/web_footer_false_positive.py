# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from yolo_v3 import non_max_suppression
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', '', 'Input directory')
tf.app.flags.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', '../models/page_based_yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def filter_inputs(X, Y):
    new_X = []
    new_Y = []

    for (x, y) in zip(X, Y):
        if np.mean(x[0:6, :, :] < 10):
            new_X.append(x)
            new_Y.append(y)

    return np.stack(new_X, axis=0), new_Y


def main(argv=None):
    np.random.seed(0)

    safe_mkdir("output/footer/")

    classes = load_coco_names(FLAGS.class_names)

    input_h = 1013
    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])
    x_min = tf.placeholder(tf.int32, shape=())
    y_min = tf.placeholder(tf.int32, shape=())

    mask_h = 20
    mask_w = input_w

    mask_val = np.zeros((mask_h, mask_w, 3), dtype=np.float32)
    mask = tf.Variable(initial_value=mask_val, dtype=tf.float32)
    padded_mask = tf.image.pad_to_bounding_box(tf.clip_by_value(mask, 0, 255), input_h-mask_h, x_min,
                                               input_h, input_w)

    black_box = np.ones_like(mask_val) 
    padded_black_box = tf.image.pad_to_bounding_box(black_box, input_h-mask_h, x_min,
                                                    input_h, input_w)

    masked_input = tf.clip_by_value(inputs * (1-padded_black_box) + padded_mask, 0, 255)

    inputs_resized = tf.image.resize_images(masked_input, (FLAGS.size, FLAGS.size),
                                            align_corners=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    detections, boxes_tensor = init_yolo(sess, inputs_resized, len(classes),
                                         FLAGS.weights_file, header_size=4)

    X_train_paths, Y_train = get_input_files_and_labels(FLAGS.input_dir + "/train/", input_h, input_w)
    X_test_paths, Y_test = get_input_files_and_labels(FLAGS.input_dir + "/test/", input_h, input_w)
    print(len(X_train_paths), len(X_test_paths))
    X_test = np.array([load_image(path) for path in X_test_paths]).astype(np.float32)

    epochs = 251
    batch_size = 4

    loss = tf.nn.relu(1.1 * FLAGS.conf_threshold - boxes_tensor[:, :, 4])
    grad = tf.gradients(tf.reduce_sum(loss), mask)[0]

    opt = tf.train.AdamOptimizer(10.0)
    grad_ph = tf.placeholder(shape=grad.get_shape().as_list(), dtype=tf.float32)
    assign_op = opt.apply_gradients([(grad_ph, mask)]) 
    sess.run(tf.variables_initializer(opt.variables()))

    assign_eps_op = tf.assign(mask, tf.clip_by_value(mask, 0, 32))

    for epoch in range(epochs):
        if epoch % 50 == 0:

            # box example: class_idx => array[([x0, y0, x1, y1]]
            # {0: [
            #       (array([1101,  581, 1400, 1007]), 1.0), 
            #       (array([ 466,  140, 1436,  389]), 1.0), 
            #       (array([1419,   25, 1540,   69]), 1.0)
            #     ]
            # }
            feed_dict = {
                inputs: X_test,
            }

            curr_loss, detected_boxes, curr_inputs = \
                batch_eval(sess, [loss, boxes_tensor, masked_input], feed_dict, extra_feed={x_min: 0, y_min: 0})

            curr_mask = sess.run(mask)

            res = Image.fromarray(curr_mask.astype(np.uint8))
            res.save('output/footer/footer_{}.png'.format(epoch))
            res.close()

            num_detect = []
            for j in range(len(X_test)):
                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                img = Image.fromarray(curr_inputs[j].astype(np.uint8))
                img.save("output/footer/img_{}_{}.png".format(epoch, j))
 
                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
                img.save("output/footer/img_boxes_{}_{}.png".format(epoch, j))

                if False:
                    my_dpi = 96
                    plt.figure(figsize=(input_h/my_dpi, input_w/my_dpi), dpi=my_dpi)
                    plt.imshow(np.array(img))
                    plt.show()

                ground_truth = 0 if 0 not in Y_test[j] else len(Y_test[j][0])
                if len(filtered_boxes) != 0:
                    num_detect.append("{}/{}".format(len(filtered_boxes[0]), ground_truth))
                else:
                    num_detect.append("{}/{}".format(0, ground_truth))
            
            print('test loss={:.3f}'.format(np.sum(curr_loss) / len(X_test)),
                  'num_boxes={}'.format(num_detect))

        batch_idx = np.random.choice(len(X_train_paths), batch_size, replace=False)
        X_batch = np.array([load_image(X_train_paths[idx]) for idx in batch_idx])
        Y_batch = [Y_train[idx] for idx in batch_idx]

        for i in range(batch_size):
            if np.random.random() > 0.75:
                h = np.random.randint(20, 100)
                c = np.random.randint(0, 255, size=1)
                X_batch[i, -(mask_h+h):-mask_h, :mask_w, :] = c

        i = 0
        start_score = 0

        max_steps = 10
        while i < max_steps:
            i += 1

            feed_dict = {
                inputs: np.clip(X_batch, 0, 255),
                y_min: 0,
                x_min: 0
            }
            curr_grad, curr_loss, detected_boxes = \
                sess.run([grad, loss, boxes_tensor], feed_dict=feed_dict)

            num_detect = [] 
            tot_detected = 0
            tot_surplus = 0

            for j in range(batch_size):

                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                if i == 1:
                    start_score = 0 if len(filtered_boxes) == 0 else len(filtered_boxes[0])

                if len(filtered_boxes) != 0:
                    num_detect.append("{}/{}".format(len(filtered_boxes[0]), len(Y_batch[j][0])))
                    tot_detected += len(filtered_boxes[0])
                    if len(filtered_boxes[0]) > len(Y_batch[j][0]) or len(filtered_boxes[0]) > start_score:
                        tot_surplus += 1
                else:
                    num_detect.append("{}/{}".format(0, len(Y_batch[j][0])))

            print(epoch, i, 'loss={:.3f}'.format(np.sum(curr_loss) / batch_size),
                  'num_boxes={}'.format(num_detect))

            if tot_surplus == batch_size:
                i = 1000
            else:
                sess.run(assign_op, feed_dict={grad_ph: curr_grad / (np.linalg.norm(curr_grad) + 1e-8)})
                sess.run(assign_eps_op)

                if i == max_steps:
                    print("no junk")


if __name__ == '__main__':
    tf.app.run()
