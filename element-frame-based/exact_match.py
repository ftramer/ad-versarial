import sys
from utils import *
import cv2
import numpy as np


def is_equal(img1, img2):
    if img1.shape != img2.shape:
        return False

    img1_flat = np.reshape(img1, -1)
    img2_flat = np.reshape(img2, -1)
    denom = max(np.linalg.norm(img1_flat), np.linalg.norm(img2_flat))
    return np.linalg.norm(img1_flat - img2_flat) / denom < 0.01


if __name__ == '__main__':
    template_path = sys.argv[1]
    glob_path = sys.argv[2]
    output_path = sys.argv[3]

    safe_mkdir(output_path)

    template_files = glob.glob(template_path + '/*.png')
    templates = []

    for file in template_files:
        img = to_alpha(cv2.imread(file, -1))
        templates.append(img)

    ad_logo_paths = get_image_paths(glob_path)
    print("found {} files to match".format(len(ad_logo_paths)))

    clusters = list(templates)
    cluster_paths = template_files

    num_matched = 0
    num_unmatched = 0

    for path in ad_logo_paths:
        img1 = to_alpha(cv2.imread(path, -1))

        matched = False
        match_idx = None

        for idx, img2 in enumerate(clusters):
            if is_equal(img1, img2):
                print("\t{} matched with {}".format(path, cluster_paths[idx]))
                matched = True
                match_idx = idx
                break

        if not matched:
            print("{} did not match! Adding new cluster".format(path))
            clusters.append(img1)
            cluster_paths.append(path)
            cv2.imwrite(output_path + "/{}.png".format(len(cluster_paths)), img1)
        elif match_idx >= len(templates):
            print("{} did not match with a template!".format(path))

        if match_idx is not None and match_idx < len(templates):
            num_matched += 1
        else:
            num_unmatched += 1

    print('number of templates: {}'.format(len(templates)))
    print('number of clusters: {}'.format(len(clusters)))
    print('num matched with template: {}'.format(num_matched))
    print('num unmatched with template: {}'.format(num_unmatched))
