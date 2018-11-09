# -*- coding: utf-8 -*-

#
# Break bbc.com/sports by adding a perturbation in the header that gets classified as an ad
#

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

    safe_mkdir("output/bbc_false_positive")

    classes = load_coco_names(FLAGS.class_names)

    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])

    mask_h = 50
    mask_w = input_w

    mask = tf.Variable(initial_value=np.zeros((mask_h, mask_w, 3)), dtype=tf.float32)
    mask_resized = mask
    mask_resized = tf.image.pad_to_bounding_box(mask_resized, 65, 0,
                                               tf.shape(inputs)[1],
                                               tf.shape(inputs)[2])
    masked_input = tf.clip_by_value(tf.add(inputs, mask_resized), 0, 255)
    inputs_resized = tf.image.resize_images(masked_input, (FLAGS.size, FLAGS.size),
                                            align_corners=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    image_files = np.array(get_images(os.path.join(FLAGS.input_dir, 'images')))

    X_train = image_files[:80]
    X_test = image_files[80:]

    eps = 4.0
    epochs = 121
    batch_size = 8

    detections, boxes_tensor = init_yolo(sess, inputs_resized, len(classes),
                                         FLAGS.weights_file, header_size=4)

    loss = tf.nn.relu(1.1 * FLAGS.conf_threshold - boxes_tensor[:, :, 4])
    grad = tf.gradients(tf.reduce_sum(loss), mask)[0]

    full_grad = tf.placeholder(dtype=tf.float32, shape=mask.shape)

    opt = tf.train.AdamOptimizer(1)
    assign_op = opt.apply_gradients([(full_grad, mask)])
    sess.run(tf.variables_initializer(opt.variables()))

    eps_assign_op = tf.assign(mask, tf.clip_by_value(mask, -eps, eps))

    for epoch in range(epochs):
        if epoch % 10 == 0:

            X_test_imgs = np.array([np.array(load_image(image_file), dtype=np.float32)
                                    for image_file in X_test])

            feed_dict = {
                inputs: X_test_imgs,
            }

            curr_loss, detected_boxes, curr_inputs = \
                batch_eval(sess, [loss, boxes_tensor, masked_input], feed_dict)
            print(detected_boxes.shape)

            curr_mask = sess.run(mask)

            res = Image.fromarray(curr_mask.astype(np.uint8))
            res.save('output/bbc_false_positive/mask_{}.png'.format(epoch))
            res.close()

            num_detect = []
            for j in range(len(X_test)):
                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                img = Image.fromarray(curr_inputs[j].astype(np.uint8))
                img.save("output/bbc_false_positive/img_{}_{}.png".format(epoch, j))

                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
                img.save("output/bbc_false_positive/img_boxes_{}_{}.png".format(epoch, j))
                img.close()

                if len(filtered_boxes) != 0:
                    num_detect.append(len(filtered_boxes[0]))

            print('test loss={:.3f}'.format(np.sum(curr_loss) / len(X_test)),
                  'num_boxes={}'.format(num_detect))

        batch_idx = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = np.array([np.array(load_image(image_file), dtype=np.float32)
                            for image_file in X_train[batch_idx]])

        i = 0
        orig_detected = [1] * batch_size

        feed_dict = {
            inputs: X_batch,
        }

        while i < 50:
            i += 1

            curr_grad, curr_loss, detected_boxes = \
                sess.run([grad, loss, boxes_tensor], feed_dict=feed_dict)


            num_detect = []

            for j in range(batch_size):

                filtered_boxes = \
                    non_max_suppression(detected_boxes[j:j+1],
                                        confidence_threshold=FLAGS.conf_threshold,
                                        iou_threshold=FLAGS.iou_threshold)

                if len(filtered_boxes) != 0:
                    num_detect.append(len(filtered_boxes[0]))
                else:
                    num_detect.append(0)

            print(epoch, i, 'loss={:.3f}'.format(np.sum(curr_loss) / batch_size),
                  'num_boxes={}/{}'.format(num_detect, batch_size))

            if np.all(np.array(num_detect) > np.array(orig_detected)):
                i = 1000

            sess.run(assign_op, feed_dict={full_grad: curr_grad / (np.linalg.norm(curr_grad) + 1e-8)})
            sess.run(eps_assign_op)


if __name__ == '__main__':
    tf.app.run()
