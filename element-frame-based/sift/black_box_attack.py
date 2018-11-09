from utils import *
import os
from sift.model import SIFTModel
import sys


def grad_estimate(SIFT, patch, orig_score, n=100, s=5, fp=False):
    g = np.zeros(patch.shape)
    for i in range(n):
        u = np.random.randn(patch.shape[0], patch.shape[1], 1)
        u = np.repeat(u, 4, axis=-1)

        temp_patch = np.clip(patch + s * u, 0, 255).astype(np.uint8)

        score, matches = SIFT.score(temp_patch, fp=fp)

        if orig_score > 1:
            p = 1 - score / orig_score
        else:
            p = 1.0
        g += p * u

        temp_patch = np.clip(patch - s * u, 0, 255).astype(np.uint8)
        score, matches = SIFT.score(temp_patch, fp=fp)

        if orig_score > 1:
            p = 1 - score / orig_score
        else:
            p = 1.0
        g -= p * u

    return g / (2*n*s)


def postprocess(SIFT, patch, patch_orig, orig_matches):

    assert(patch.shape == patch_orig.shape)
    patch = patch.astype(np.uint8)

    norm = np.linalg.norm(
        (patch.astype(np.float32) - patch_orig.astype(np.float32)).reshape(-1)
    )
    print("norm before: {}".format(norm))

    h, w, _ = patch.shape
    for i in range(h):
        for j in range(w):
            old = patch[i, j, :].copy()
            patch[i, j, :] = patch_orig[i, j, :].copy()

            score, matches = SIFT.score(patch.astype(np.uint8))

            if matches > orig_matches:
                patch[i, j, :] = old.copy()

    norm = np.linalg.norm(
        (patch.astype(np.float32) - patch_orig.astype(np.float32)).reshape(-1))
    print("norm after: {}".format(norm))

    return patch


def PGD(SIFT, patch, n=1000, path=None):
    patch_adv = patch.copy().astype(np.float32)
    patch_orig = patch.copy().astype(np.float32)

    alpha = 15000.0 / n * 5.0

    orig_score, orig_matches = SIFT.score(patch_adv.astype(np.uint8))
    prev_score = orig_score
    prev_matches = orig_matches
    prev_adv = patch_adv.copy()

    i = 0
    num_rewinds = 0

    target = 0
    print("target matches: {}".format(target))

    norm_alpha = 0.01

    while i < n:

        curr_score, curr_matches = SIFT.score(patch_adv.astype(np.uint8))

        if curr_matches <= target:
            return patch_adv.astype(np.uint8)
        norm = np.linalg.norm((patch_adv - patch_orig).reshape(-1))
        print("PGD step {}: score={}, matches={}, norm={}".format(
            i, curr_score, curr_matches, norm)
        )

        if (curr_matches >= prev_matches and
                curr_score >= prev_score):
            num_rewinds += 1
            print("rewind %d ..." % num_rewinds)

            if num_rewinds >= 10:
                norm_alpha /= 2
                num_rewinds = 0
            else:
                patch_adv = prev_adv.copy()
        else:
            i += 1
            prev_matches = curr_matches
            prev_adv = patch_adv.copy()
            prev_score = curr_score
            num_rewinds = 0

        g = grad_estimate(SIFT, patch_adv, curr_score, n=100, s=5)

        norm = np.linalg.norm(np.reshape(g, -1))
        g_norm = g / norm
        g_norm = alpha * g_norm

        patch_adv += (g_norm + norm_alpha * (patch_orig - patch_adv))
        patch_adv = np.clip(patch_adv, 0, 255)

        if i % 10 == 0 and num_rewinds == 0:

            score, num_match = SIFT.score(patch_adv.astype(np.uint8))

            print(score, num_match)
            cv2.imwrite(path, patch_adv.astype(np.uint8))
            patch_adv = patch_adv.astype(np.uint8).astype(np.float32)

    return patch_adv.astype(np.uint8)


if __name__ == '__main__':
    np.random.seed(1)

    cv2_sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()

    src_logo_file = sys.argv[1]
    output_dir = sys.argv[2]

    safe_mkdir(output_dir)

    fname = os.path.basename(src_logo_file).split('.')[0]

    template_path = sys.argv[3]
    match_threshold = 0.05
    match_threshold_small = 1.0
    SIFT = SIFTModel(template_path, match_threshold=match_threshold,
                     match_threshold_small=match_threshold_small)

    src_logo = cv2.imread(src_logo_file, -1)

    patch_adv = src_logo.copy()

    score, num_match = SIFT.score(patch_adv)
    print(fname, score, num_match)

    patch_adv = PGD(SIFT, patch_adv, path=output_dir + "/img_PGD_{}.png".format(fname))

    score, num_match = SIFT.score(patch_adv)
    print(fname, score, num_match)

    cv2.imwrite(output_dir + "/img_PGD_{}.png".format(fname), patch_adv)

    print('postprocess...')
    patch_adv = postprocess(SIFT, patch_adv, src_logo, num_match)

    score, num_match = SIFT.score(patch_adv)
    print(fname, score, num_match)

    cv2.imwrite(output_dir + "/img_PGD_{}_post.png".format(fname), patch_adv)

    img = src_logo
    if has_alpha(img):
        img = blend_white(img).astype(np.float32)
    else:
        img = img.astype(np.float32)

    img /= 255.0

    adv_img = patch_adv
    if has_alpha(adv_img):
        adv_img = blend_white(adv_img).astype(np.float32)
    else:
        adv_img = adv_img.astype(np.float32)

    adv_img /= 255.0

    diff = np.linalg.norm((img - adv_img).reshape(-1))

    with open(output_dir + '/{}_norm.txt'.format(fname), 'w') as f:
        f.write("{:3f}".format(diff))
