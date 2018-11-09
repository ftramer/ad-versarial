import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image
from yolo_v3 import yolo_v3, load_weights, detections_boxes
import os
import glob
import cv2


def batch_eval(sess, tf_outputs, feed_dict, batch_size=None, extra_feed=None):

    vals = [np.asarray(v) for v in feed_dict.values()]
    return batch_eval_ch(sess, list(feed_dict.keys()), tf_outputs, vals, batch_size=batch_size, feed=extra_feed)


# adapted from Cleverhans
def batch_eval_ch(sess, tf_inputs, tf_outputs, numpy_inputs, batch_size=None, feed=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    This version uses exactly the tensorflow graph constructed by the
    caller, so the caller can place specific ops on specific devices
    to implement model parallelism.
    Most users probably prefer `batch_eval_multi_worker` which maps
    a single-device expression to multiple devices in order to evaluate
    faster by parallelizing across data.
    :param sess: tf Session to use
    :param tf_inputs: list of tf Placeholders to feed from the dataset
    :param tf_outputs: list of tf tensors to calculate
    :param numpy_inputs: list of numpy arrays defining the dataset
    :param batch_size: int, batch size to use for evaluation
      If not specified, this function will try to guess the batch size,
      but might get an out of memory error or run the model with an
      unsupported batch size, etc.
    :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
    """

    if batch_size is None:
        batch_size = 8

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    for start in range(0, m, batch_size):
        batch = start // batch_size

        # Compute batch start and end indices
        start = batch * batch_size
        end = start + batch_size
        numpy_input_batches = [numpy_input[start:end]
                               for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= batch_size
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        feed_dict = dict(zip(tf_inputs, numpy_input_batches))
        if feed is not None:
            feed_dict.update(feed)
        numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(image_file):
    #pil_image = Image.open(image_file)
    #img = np.array(pil_image, dtype=np.float32)
    #pil_image.close()

    img = np.asarray(cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB), dtype=np.float32)
    return img


def get_input_files_and_labels(input_dir, input_h, input_w):
    image_files = get_images(os.path.join(input_dir, 'images'))
    image_names = [get_file_name(f) for f in image_files]
    label_files = [os.path.join(input_dir, 'labels', name + '.txt')
                   for name in image_names]
    assert np.all([os.path.isfile(lf) for lf in label_files])

    all_labels = np.array([load_labels(label_file)
                           for label_file in label_files])
    all_labels = [convert_labels(labels, (input_w, input_h)) for labels in all_labels]

    return np.array(image_files), all_labels


def load_full_pages(input_dir, input_h, input_w):
    image_files = get_images(input_dir)
    image_names = [get_file_name(f) for f in image_files]

    images = []

    for image_file in image_files:
        img = Image.open(image_file)
        w, h = img.size
        ratio = w / (1.0 * input_w)
        new_h = int(h * ratio)
        img = np.array(img.resize((input_w, new_h)))[:input_h, :, :].astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)

    return np.array(images), image_names


def convert_labels(boxes, img_size):
    result = {}
    for cls, bboxs in boxes.items():
        for box, score in bboxs:
            x, y, w, h = box
            mid_x = x * img_size[0]
            mid_y = y * img_size[1]

            x0 = mid_x - w / 2 * img_size[0]
            x1 = mid_x + w / 2 * img_size[0]

            y0 = mid_y - h / 2 * img_size[1]
            y1 = mid_y + h / 2 * img_size[1]

            new_box = np.array([int(x0), int(y0), int(x1), int(y1)])
            if cls not in result:
                result[cls] = []
            result[cls].append((new_box, score))
    return result


def parse_labels(line):
    vals = line.split(' ')
    return int(vals[0]), \
           np.array([float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])])


def load_labels(label_file):
    with open(label_file) as inf:
        lines = inf.readlines()

        result = {}
        for line in lines:
            cls, box = parse_labels(line)
            if cls not in result:
                result[cls] = []
            result[cls].append((box, 1.0))
        return result


def get_images(dir_path):
    image_files = []
    for ext in ('*.png', '*.jpg'):
        image_files.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(image_files)


def get_file_name(f):
    return os.path.splitext(os.path.basename(f))[0]


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple([255, 0, 0])
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            cor = box
            line = (cor[0], cor[1], cor[0], cor[3])
            draw.line(line, fill=color, width=10)
            line = (cor[0], cor[1], cor[2], cor[1])
            draw.line(line, fill=color, width=10)
            line = (cor[0], cor[3], cor[2], cor[3])
            draw.line(line, fill=color, width=10)
            line = (cor[2], cor[1], cor[2], cor[3])
            draw.line(line, fill=color, width=10)

            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=tuple([0, 0, 255]))


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def init_yolo(sess, inputs, num_classes, weights, header_size=5):
    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, num_classes, data_format='NHWC')
        load_ops = load_weights(tf.global_variables(scope='detector'), weights, header_size=header_size)

    boxes = detections_boxes(detections)
    sess.run(load_ops)
    return detections, boxes
