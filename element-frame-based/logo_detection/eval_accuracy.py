from .yolo_v3 import backbone
import logo_detection.yolo_utils as yolo_utils
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from utils import *
import os
from logo_detection import CKPT_PATH, NAMES_PATH


adv_output_dir = "logo_detection/output/"
test_images_path = "../data/web/all_frames.txt"


def preprocess(img):
    img = to_bgr(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return img_float


def preprocess_tf(inputs):
    inputs_resized = tf.image.resize_images(inputs, (416, 416),
                                            align_corners=True)

    inputs_scaled = inputs_resized / 255.
    return inputs_scaled


def has_object(image_file):
    label_file = os.path.splitext(image_file)[0] + ".txt"
    with open(label_file) as f:
        return len(f.readlines()) > 0


def batch_generator(file_list, batch_size=16):
    for i in range(0, len(file_list), batch_size):
        paths = file_list[i:i+batch_size]
        images = [preprocess(cv2.imread(img_path)) for img_path in paths]
        labels = [has_object(img_path) for img_path in paths]

        images = np.concatenate(images, axis=0).astype(np.float32)
        yield images, labels, paths


class PGD(object):

    def __init__(self, boxes, inputs, eps=4.0, k=40):
        self.confidence_threshold = 0.5

        self.inputs = inputs
        self.boxes = boxes
        self.loss_evade = tf.reduce_sum(tf.nn.relu(boxes[:, :, 4] - 0.9 * self.confidence_threshold))
        self.loss_generate = -tf.reduce_sum(boxes[:, :, 4])

        self.grad_evade = tf.gradients(self.loss_evade, self.inputs)[0]
        self.grad_generate = tf.gradients(self.loss_generate, self.inputs)[0]

        self.eps = eps
        self.alpha = max(2 * eps / k, 0.25)
        self.k = k

    def attack(self, orig_img, gen_boxes=False):

        adv_img = orig_img.copy()

        grad = self.grad_evade
        loss = self.loss_evade

        if gen_boxes:
            print("generation attack!")
            grad = self.grad_generate
            loss = self.loss_generate
            size_y = np.random.randint(40, 200)
            size_x = np.random.randint(40, 200)
            print(size_y, size_x)
            adv_img = 255 * np.ones((1, size_y, size_x, 3))
            orig_img = adv_img.copy()

        for i in range(self.k):
            grad_np, curr_loss, curr_boxes = \
                sess.run([grad, loss, self.boxes],
                         feed_dict={inputs: adv_img})

            curr_boxes = yolo_utils.non_max_suppression(curr_boxes)[0]
            print(i, curr_loss, len(curr_boxes))

            if (gen_boxes and len(curr_boxes) > 0) or (not gen_boxes and len(curr_boxes) == 0):
                return adv_img

            if not gen_boxes:
                grad_np[:, :15, :, :] = 0

            adv_img -= self.alpha * np.sign(grad_np)
            adv_img = np.clip(adv_img, 0, 255)
            adv_img = np.clip(adv_img, orig_img - self.eps, orig_img + self.eps)

        return adv_img


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])
    inputs_resized = preprocess_tf(inputs)
    predictions = backbone(inputs_resized, 1, is_training=False, scope='yolov3')
    predictions = yolo_utils.detections_boxes(predictions)

    # load weights
    saver = yolo_utils.restore_saver()
    names = yolo_utils.load_coco_name(NAMES_PATH)

    with open(test_images_path, 'r') as f:
        test_images = [line.rstrip() for line in f.readlines()]

    test_images = test_images[:100]

    y_true = []
    y_pred = []
    y_pred_adv = []
    total = 0

    safe_mkdir(adv_output_dir)

    with tf.Session() as sess:
        saver.restore(sess, CKPT_PATH)

        pgd = PGD(predictions, inputs, eps=4.0, k=200)

        for (images, labels, paths) in batch_generator(test_images, batch_size=1):
            y_batch = sess.run(predictions, feed_dict={inputs: images})
            y_batch = yolo_utils.non_max_suppression(y_batch)

            y_true.extend(labels)
            y_pred.extend([len(d) > 0 for d in y_batch])
            total += len(images)

            print("y_true: {}".format(labels))
            print("y_pred: {}".format([len(d) > 0 for d in y_batch]))

            if labels[-1] != y_pred[-1]:
                print("error on {}. Should be {}".format(paths[-1], labels[-1]))

            #if y_true[-1] != y_pred[-1]:
            #    visual.vis(images.astype(np.uint8), y_batch, (416, 416), names, None, convert_rgb=False)

            images_adv = pgd.attack(images, labels[0] == 0)
            y_batch_adv = sess.run(predictions, feed_dict={inputs: images_adv})
            y_batch_adv = yolo_utils.non_max_suppression(y_batch_adv)
            y_pred_adv.extend([len(d) > 0 for d in y_batch_adv])
            print("y_pred_adv: {}".format([len(d) > 0 for d in y_batch_adv]))

            cv2.imwrite(adv_output_dir + "/{}.png".format(total),
                        cv2.cvtColor(images_adv[0].astype(np.uint8), cv2.COLOR_RGB2BGR))

            print(total)

        print("evaluated {} images".format(total))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("num ads: {}".format(np.sum(y_true)))
        print("accuracy: {:.1f}%".format(100.0 * (tp + tn) / (1.0 * total)))
        print("precision: {:.1f}%".format(100.0 * tp / (1.0 * (tp + fp))))
        print("recall: {:.1f}%".format(100.0 * tp / (1.0 * (tp + fn))))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_adv).ravel()
        print("accuracy (adv): {:.1f}%".format(100.0 * (tp + tn) / (1.0 * total)))
        print("precision (adv): {:.1f}%".format(100.0 * tp / (1.0 * (tp + fp))))
        print("recall (adv): {:.1f}%".format(100.0 * tp / (1.0 * (tp + fn))))
