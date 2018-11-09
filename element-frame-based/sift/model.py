import cv2
from utils import to_gray, has_alpha, blend_white, get_image_paths
import numpy as np
import glob
import sys
from timeit import default_timer as timer


def resize(patch):
    target_height = 15 * 8
    curr_height = patch.shape[0]
    mul = target_height / (1.0 * curr_height)

    return cv2.resize(patch, None, fx=mul, fy=mul)


def preproc(patch):
    patch = resize(patch)
    patch = to_gray(patch)
    return patch


def get_keypoints(cv2_sift, template):
    return cv2_sift.detectAndCompute(preproc(template), None)


def sift_pred(cv2_sift, bf, query_kp, query_des, patch,
              patch_kp=None, patch_des=None,
              template_img=None, draw_matches=False, ratio=0.6, fp=False):

    if patch_kp is None or patch_des is None:
        patch_kp, patch_des = get_keypoints(cv2_sift, patch)

    if patch_des is None:
        match_list = []
    else:
        match_list = bf.knnMatch(query_des, patch_des, k=2)
        match_list = [m for m in match_list if len(m) == 2]

    # Apply ratio test
    good = []
    score = 0.0
    for m, n in match_list:
        if m.distance < ratio * n.distance:
            good.append([m])
            if not fp:
                score += n.distance / np.maximum(m.distance, 0.01)
        else:
            if fp:
                score += np.sqrt((m.distance / n.distance - ratio))

    if draw_matches:
        template_img = resize(template_img.copy())
        if has_alpha(template_img):
            template_img = blend_white(template_img)
        if has_alpha(patch):
            patch = blend_white(patch)

        drawn_matches = cv2.drawMatchesKnn(template_img,
                                           query_kp,
                                           resize(patch),
                                           patch_kp,
                                           good, None, flags=2)

        return score, len(good), drawn_matches

    return score, len(good)


class SIFTModel(object):

    def __init__(self, path_to_templates,
                 match_threshold=0.2, match_threshold_small=0.5):
        self.cv2_sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.match_threshold = match_threshold
        self.match_threshold_small = match_threshold_small

        templates, keypoints = self.load_templates(path_to_templates)
        self.templates = templates
        self.keypoints = keypoints

    def load_templates(self, path_to_templates):
        print("LOADING TEMPLATES...")

        if path_to_templates.endswith(".png"):
            files = [path_to_templates]
        else:
            files = glob.glob(path_to_templates + '/*.png')

        templates = []
        keypoints = []

        for file in files:
            img = cv2.imread(file, -1)
            kp, des = get_keypoints(self.cv2_sift, img)

            if len(kp) > 10:
                templates.append(file)
                keypoints.append((kp, des))
                print("adding {} as template with {} keypoints".format(file, len(kp)))
            else:
                print("ignoring {}".format(file))

        print("LOADED {} TEMPLATES".format(len(templates)))
        return templates, keypoints

    def score(self, img, fp=False):
        patch_kp, patch_des = get_keypoints(self.cv2_sift, img)

        total_score = 0.0
        max_matches = 0

        for idx, (kp, des) in enumerate(self.keypoints):
            score, num_matches = sift_pred(self.cv2_sift, self.bf, kp, des,
                                           patch=img,
                                           patch_kp=patch_kp,
                                           patch_des=patch_des, fp=fp)

            if fp:
                total_score += score
                max_matches = max(max_matches, num_matches)
            else:
                if len(kp) > 40:
                    threshold = self.match_threshold
                else:
                    threshold = self.match_threshold_small

                match_ratio = (1.0 * num_matches) / len(kp)

                if match_ratio >= threshold:
                    total_score += score
                    max_matches = max(max_matches, num_matches)

        return total_score, max_matches

    def match(self, img, verbose=False, img_name=None):
        patch_kp, patch_des = get_keypoints(self.cv2_sift, img)

        high_match = np.array([0, 0, 0])
        for idx, (kp, des) in enumerate(self.keypoints):
            _, num_matches = sift_pred(self.cv2_sift, self.bf, kp, des,
                                       patch=img,
                                       patch_kp=patch_kp,
                                       patch_des=patch_des)

            if len(kp) > 40:
                threshold = self.match_threshold
            else:
                threshold = self.match_threshold_small

            match_ratio = (1.0 * num_matches) / len(kp)

            if match_ratio > high_match[0]:
                high_match = [match_ratio, num_matches, len(kp)]
            if match_ratio >= threshold:
                if verbose:
                    print("{} matched on template {} with {}/{}={:.2f} "
                          "matches".format(img_name, self.templates[idx],
                                           num_matches, len(kp), match_ratio))
                return True, high_match

        if verbose:
            print("\tno match for {}! Highest score was {}".format(
                img_name, high_match))
        return False, high_match


if __name__ == '__main__':
    template_path = sys.argv[1]
    glob_path = sys.argv[2]

    match_threshold = 0.05
    match_threshold_small = 1.0
    SIFT = SIFTModel(template_path, match_threshold=match_threshold,
                     match_threshold_small=match_threshold_small)

    ad_logo_paths = get_image_paths(glob_path)
    print("found {} files to match".format(len(ad_logo_paths)))

    scores = []

    t1 = timer()
    for ad_logo_path in ad_logo_paths:
        ad_logo = cv2.imread(ad_logo_path, -1)
        assert ad_logo is not None

        if ad_logo.dtype != np.uint8:
            assert ad_logo.dtype == np.uint16
            ad_logo = (ad_logo / 256).astype('uint8')

        try:
            match, (match_ratio, _, _) = SIFT.match(ad_logo, verbose=True,
                                                    img_name=ad_logo_path)

            if not match:
                match_ratio = 0
            scores.append(match_ratio)
        except Exception as e:
            print("{} failed with {}".format(ad_logo_path, e))

    t2 = timer()
    print("evaluated {} images in {} seconds".format(len(ad_logo_paths), t2 - t1))

    scores = np.array(scores)
    ad_logo_paths = np.array(ad_logo_paths)

    print(np.sum(scores >= match_threshold))
    print(ad_logo_paths[scores >= match_threshold])
    topk = scores.argsort()[-10:][::-1]
    print(ad_logo_paths[topk])

