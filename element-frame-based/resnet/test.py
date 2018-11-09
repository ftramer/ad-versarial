import keras
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from utils import *
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer

import tensorflow as tf

test_images_path = "../data/web/all_frames.txt"
all_ads_path = "../data/web/all_ads.txt"
adv_output_dir = "resnet/output/"
model_path = "../external/keras_resnet.h5"


def preprocess(img):
    img = to_bgr(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    return img_float


def preprocess_tf(inputs):
    inputs_resized = tf.image.resize_images(inputs, (256, 256),
                                            align_corners=True)
    return inputs_resized


class LinfPGDAttack:
    def __init__(self, x, y, preds, epsilon, k, a, random_start=False, clip_min=0, clip_max=255):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.x = x
        self.y = y

        self.preds = preds
        self.loss = -tf.reduce_mean(tf.reduce_sum(preds * y, axis=-1))
        self.grad = tf.gradients(self.loss, x)[0]

    def perturb(self, x_nat, y, sess):

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad, loss = sess.run([self.grad, self.loss],
                                  feed_dict={self.x: x, self.y: y})

            print("iter {}, loss = {:.3f}".format(i, loss))
            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, self.clip_min, self.clip_max) # ensure valid pixel range

        return x


def batch_generator(file_list, all_ads, batch_size=16):
    for i in range(0, len(file_list), batch_size):
        paths = file_list[i:i+batch_size]
        labels = [f in all_ads for f in paths]

        images = [preprocess(cv2.imread(img_path)) for img_path in paths]
        images = np.concatenate(images, axis=0).astype(np.float32)
        yield images, labels, paths


if __name__ == '__main__':

    config = tf.ConfigProto(
        #device_count = {'GPU': 0},
        #intra_op_parallelism_threads=1,
        #inter_op_parallelism_threads=1
    )

    safe_mkdir(adv_output_dir)

    with tf.Session(config=config) as sess:
        keras.backend.set_session(sess)
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        inputs = tf.placeholder(tf.float32, [None, None, None, 3])
        inputs_resized = preprocess_tf(inputs)
        y = tf.placeholder(tf.float32, shape=(None, 2))

        # Invert the classes. The ResNet predicts class 0 for ads
        preds = 1. - model(inputs_resized)

        with open(test_images_path, 'r') as f:
            test_images = [line.rstrip() for line in f.readlines()]

        with open(all_ads_path, 'r') as f:
            all_ads = [line.rstrip() for line in f.readlines()]

        y_true = []
        y_pred = []
        y_pred_adv = []
        total = 0

        eps = 2.0
        max_iters = 200
        pgd = LinfPGDAttack(inputs, y, preds, epsilon=eps, k=max_iters, a=10*eps/max_iters)

        for (images, labels, paths) in batch_generator(test_images, all_ads, batch_size=1):

            t1 = timer()
            y_batch = sess.run(preds, feed_dict={inputs: images})
            t2 = timer()
            print("\tinference time: {}".format(t2-t1))

            y_true.extend(labels)
            y_pred.extend(np.argmax(y_batch, axis=-1))
            total += len(images)

            print(paths)
            print("y_true: {}".format(labels))
            print("y_pred: {}".format(np.argmax(y_batch, axis=-1)))

            labels_one_hot = to_categorical(labels, num_classes=2)

            if not labels[0]:
                size_x = np.random.randint(20, 200)
                size_y = np.random.randint(20, 200)
                print(size_x, size_y)
                images = 127 * np.ones((1, size_y, size_x, 3)).astype(np.float32)
                eps = 2.0
            else:
                eps = 2.0

            images_adv = pgd.perturb(images, labels_one_hot, sess)
            y_batch_adv = sess.run(preds, feed_dict={inputs: images_adv})

            y_pred_adv.extend(np.argmax(y_batch_adv, axis=-1))

            cv2.imwrite(adv_output_dir + "/{}.png".format(total),
                        cv2.cvtColor(images_adv[0].astype(np.uint8), cv2.COLOR_RGB2BGR))

            print(total)

        print("evaluated {} images".format(total))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("accuracy: {:.1f}%".format(100.0 * (tp + tn) / (1.0 * total)))
        print("precision: {:.1f}%".format(100.0 * tp / (1.0 * (tp + fp))))
        print("recall: {:.1f}%".format(100.0 * tp / (1.0 * (tp + fn))))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_adv).ravel()
        print("accuracy (adv): {:.1f}%".format(100.0 * (tp + tn) / (1.0 * total)))
        print("precision (adv): {:.1f}%".format(100.0 * tp / (1.0 * (tp + fp))))
        print("recall (adv): {:.1f}%".format(100.0 * tp / (1.0 * (tp + fn))))
