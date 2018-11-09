# -*- coding: utf-8 -*-

from yolo_v3 import non_max_suppression
from utils import *
import os
from timeit import default_timer as timer


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', '', 'Input directory')
tf.app.flags.DEFINE_string('output_dir', '', 'Output directory')
tf.app.flags.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', '../models/page_based_yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def main(argv=None):
    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    config = tf.ConfigProto(
        #device_count={'GPU': 0},
        #intra_op_parallelism_threads=1,
        #inter_op_parallelism_threads=1
    )
    sess = tf.Session(config=config)
    detections, boxes = init_yolo(sess, inputs, len(classes), FLAGS.weights_file, header_size=4)

    image_files = get_images(os.path.join(FLAGS.input_dir, 'images'))
    image_names = [get_file_name(f) for f in image_files]
    label_files = [os.path.join(FLAGS.input_dir, 'labels', name + '.txt') for name in image_names]
    assert np.all([os.path.isfile(lf) for lf in label_files])

    safe_mkdir(FLAGS.output_dir)

    for idx, image_file in enumerate(image_files):
        print(image_file)
        img_orig = Image.open(image_file)
        img = img_orig.resize((416, 416))

        t1 = timer()        
        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img, dtype=np.float32)]})
        t2 = timer()

        filtered_boxes = non_max_suppression(detected_boxes, 
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)
        t3 = timer()
        print("\tinference time: {}".format(t2-t1))
        print("\ttotal time: {}".format(t3-t1))

        draw_boxes(filtered_boxes, img_orig, classes, (FLAGS.size, FLAGS.size))
        img_orig.save(FLAGS.output_dir + "/{}.png".format(idx))


if __name__ == '__main__':
    tf.app.run()
