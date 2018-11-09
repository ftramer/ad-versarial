
from yolo_v3 import non_max_suppression
from utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('full_page', '', 'Webpage to test scrolling on')
tf.app.flags.DEFINE_string('mask', '', 'Mask')
tf.app.flags.DEFINE_string('footer', '', 'Footer')
tf.app.flags.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', '../models/page_based_yolov3.weights', 'Binary file with detector weights')


tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def main(argv=None):

    safe_mkdir("output/scroll/")

    np.random.seed(0)
    classes = load_coco_names(FLAGS.class_names)

    input_h = 1013
    input_w = 1919
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])
    inputs_resized = tf.image.resize_images(inputs, (FLAGS.size, FLAGS.size), align_corners=True)

    if FLAGS.mask:
        alpha = tf.placeholder(shape=(2, 1, 1, 1), dtype=tf.float32)
        mask_tile = 8

        mask_val = load_image(FLAGS.mask)

        mask = tf.Variable(initial_value=mask_val, dtype=tf.float32)
        slack_h = input_h - mask_val.shape[0] * mask_tile
        slack_w = input_w - mask_val.shape[1] * mask_tile
        tiled_mask = tf.image.pad_to_bounding_box(tf.tile(mask, [mask_tile, mask_tile, 1]), slack_h // 2, slack_w // 2,
                                                  input_h, input_w)
        masked_input = tf.clip_by_value((1 - alpha) * inputs + alpha * tiled_mask, 0, 255)
        inputs_resized = tf.image.resize_images(masked_input, (FLAGS.size, FLAGS.size),
                                                align_corners=True)

    if FLAGS.footer:
        footer = load_image(FLAGS.footer)
        footer_h = footer.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    detections, boxes_tensor = init_yolo(sess, inputs_resized, len(classes),
                                         FLAGS.weights_file, header_size=4)

    full_page = Image.open(FLAGS.full_page)
    w, h = full_page.size

    ratio = w / (1.0 * input_w)
    new_h = int(h * ratio)
    full_page = np.array(full_page.resize((input_w, new_h))).astype(np.float32)
    full_page = cv2.cvtColor(full_page, cv2.COLOR_RGBA2RGB)

    to_scroll = new_h - input_h
    print(to_scroll)

    num_outputs = 100
    scroll_dh = to_scroll // num_outputs

    for i in range(num_outputs):
        img = full_page[i*scroll_dh:i*scroll_dh + input_h, :, :].astype(np.float32)
        img_adv = img.copy().astype(np.float32)

        feed_dict = {
            inputs: [img, img_adv]
        }

        if FLAGS.footer:
            img_adv[-footer_h:, :, :] = footer

        if FLAGS.mask:
            feed_dict[alpha] = [[[[0.0]]], [[[0.01]]]]

        detected_boxes = sess.run(boxes_tensor, feed_dict=feed_dict)

        filtered_boxes = non_max_suppression(detected_boxes[:1], confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)

        filtered_boxes_adv = non_max_suppression(detected_boxes[1:], confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)

        num_ads = 0 if len(filtered_boxes) == 0 else len(filtered_boxes[0])
        num_ads_adv = 0 if len(filtered_boxes_adv) == 0 else len(filtered_boxes_adv[0])

        print(i, num_ads, num_ads_adv)

        img = Image.fromarray(img.astype(np.uint8))
        draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
        img.save("output/scroll/img_boxes_{0:03d}.png".format(i))

        img_adv = Image.fromarray(img_adv.astype(np.uint8))
        draw_boxes(filtered_boxes_adv, img_adv, classes, (FLAGS.size, FLAGS.size))
        img_adv.save("output/scroll/img_adv_boxes_{0:03d}.png".format(i))


if __name__ == '__main__':
    tf.app.run()
