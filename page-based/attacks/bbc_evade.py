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


def main(argv=None):
    np.random.seed(0)

    safe_mkdir("output/bbc_evade")

    classes = load_coco_names(FLAGS.class_names)

    input_h = 1013
    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])

    x_min = tf.placeholder(tf.int32, shape=[None])
    y_min = tf.placeholder(tf.int32, shape=[None])
    x_min2 = tf.placeholder(tf.int32, shape=[None])
    y_min2 = tf.placeholder(tf.int32, shape=[None])

    mask_h = 40
    mask_w = 820 + 200
    mask = tf.Variable(initial_value=255 + 0*np.random.randint(low=0, high=255, size=(mask_h, mask_w, 3)), dtype=tf.float32)

    padded_mask = tf.map_fn(lambda dims: tf.image.pad_to_bounding_box(mask, dims[0], dims[1],
                                               tf.shape(inputs)[1],
                                               tf.shape(inputs)[2]),
                            (y_min, x_min), dtype=tf.float32)
 
    black_box = tf.ones([mask_h, mask_w, 3], dtype=tf.float32)
    black_mask = 1.0 - tf.map_fn(lambda dims: tf.image.pad_to_bounding_box(black_box, dims[0], dims[1],
                                                    tf.shape(inputs)[1],
                                                    tf.shape(inputs)[2]),
                            (y_min, x_min), dtype=tf.float32)

    blacked_inputs = tf.multiply(inputs, black_mask)
    masked_input = tf.clip_by_value(tf.add(blacked_inputs, padded_mask), 0, 255)
    inputs_resized = tf.image.resize_images(masked_input, (FLAGS.size, FLAGS.size),
                                            align_corners=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    image_files = np.array(get_images(os.path.join(FLAGS.input_dir, 'images')))
    np.random.seed(0)
    np.random.shuffle(image_files)
    image_names = [get_file_name(f) for f in image_files]

    label_files = [os.path.join(FLAGS.input_dir, 'labels', name + '.txt')
                   for name in image_names]
    assert np.all([os.path.isfile(lf) for lf in label_files])

    all_labels = np.array([load_labels(label_file)
                           for label_file in label_files])

    all_labels = [convert_labels(labels, (input_w, input_h)) for labels in all_labels]

    X_train = image_files[:80]
    Y_train = all_labels[:80]
    X_test = image_files[80:]
    Y_test = all_labels[80:]

    X_test = np.array([np.array(load_image(image_file), dtype=np.float32)
                       for image_file in X_test])

    print(len(X_train), len(X_test))
    epochs = 151
    batch_size = 8

    detections, boxes_tensor = init_yolo(sess, inputs_resized, len(classes),
                                         FLAGS.weights_file, header_size=4)

    loss = tf.reduce_sum(tf.nn.relu(boxes_tensor[:, :, 4] - 0.1 * FLAGS.conf_threshold))
    grad = tf.gradients(loss, mask)[0]

    full_grad = tf.placeholder(dtype=tf.float32, shape=mask.shape)

    eps = 3.0

    opt = tf.train.AdamOptimizer(1.0)
    assign_op = opt.apply_gradients([(full_grad, mask)])
    sess.run(tf.variables_initializer(opt.variables()))

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print_idx = (epoch // 10) % len(X_test)

            boxes = [label[0][0][0] for label in Y_test]
            x0 = [box[0] + 5 - 100 for box in boxes]
            y1 = [box[-1] - 10 for box in boxes]
            y0 = [box[1] + 5 for box in boxes]
            feed_dict = {
                inputs: X_test,
                x_min: x0,
                y_min: y1,
                x_min2: x0,
                y_min2: y0
            }

            detected_boxes, curr_inputs = batch_eval(sess, [boxes_tensor, masked_input], feed_dict)

            num_detect = []
            for j in range(len(X_test)):
                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                img = Image.fromarray(curr_inputs[j].astype(np.uint8))
                img.save("output/bbc_evade/img_{}_{}.png".format(epoch, j))

                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
                img.save("output/bbc_evade/img_boxes_{}_{}.png".format(epoch, j))
                
                if False:
                    img_masked = sess.run(masked_input, feed_dict=feed_dict)
                    img = Image.fromarray(img_masked[print_idx].astype(np.uint8))
                    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
                    my_dpi = 96
                    plt.figure(figsize=(input_h/my_dpi, input_w/my_dpi), dpi=my_dpi)
                    plt.imshow(np.array(img))
                    plt.show()

                if len(filtered_boxes) > 0:
                    num_detect.append(len(filtered_boxes[0]))
                else:
                    num_detect.append(0)
            
            print('num_boxes={}'.format(num_detect))

        batch_idx = np.random.choice(len(X_train), batch_size, replace=False)

        X_batch = np.array([np.array(load_image(image_file), dtype=np.float32)
                           for image_file in X_train[batch_idx]])
        Y_batch = [Y_train[i] for i in batch_idx]

        ad_idx = np.random.choice(len(X_batch), batch_size, replace=True)
        for i in range(batch_size):
            x0, y0, x1, y1 = Y_batch[ad_idx[i]][0][0][0]
            ad = X_batch[ad_idx[i], y0:y1, x0:x1, :]
            x0b, y0b, x1b, y1b = Y_batch[i][0][0][0]
            
            x1b = x0b + (x1-x0)
            y1b = y0b + (y1-y0)           
 
            X_batch[i, y0b:y1b, x0b:x1b, :] = ad

        boxes = [label[0][0][0] for label in Y_batch] 
        x0 = [box[0] + 5 - 100 for box in boxes]
        y1 = [box[-1] - 10 for box in boxes]
        y0 = [box[1] + 5 for box in boxes]

        i = 0
        stop = False
        while not stop:
            i += 1

            feed_dict = {
                inputs: X_batch,
                x_min: x0,
                y_min: y1,
                x_min2: x0,
                y_min2: y0
            }
            curr_grad, curr_loss, detected_boxes = \
                sess.run([grad, loss, boxes_tensor], feed_dict=feed_dict)

            num_detect = 0

            for j in range(batch_size):

                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                if len(filtered_boxes) != 0:
                    num_detect += len(filtered_boxes[0])

            print(epoch, i, 'loss={:.3f}'.format(curr_loss / batch_size),
                  'num_boxes={}/{}'.format(num_detect, batch_size))

            if (num_detect == 0) or (i >= 50):
                stop = True

            sess.run(assign_op, feed_dict={full_grad: curr_grad / (np.linalg.norm(curr_grad) + 1e-8)})
            sess.run(tf.assign(mask, tf.clip_by_value(mask, 255-eps, 255)))


if __name__ == '__main__':
    tf.app.run()
