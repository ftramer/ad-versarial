from utils import *
import sys
from sift.model import SIFTModel
from sift.black_box_attack import grad_estimate


def PGD(SIFT, patch, n=5000, path=None):
    patch_adv = patch.copy().astype(np.float32)
    patch_orig = patch.copy().astype(np.float32)

    lr = 2500

    orig_score, orig_matches = SIFT.score(patch_adv.astype(np.uint8), fp=True)
    best_matches = orig_matches

    i = 0
    num_rewinds = 0

    while i < n:
        curr_score, curr_matches = SIFT.score(patch_adv.astype(np.uint8), fp=True)

        norm = np.linalg.norm((patch_adv - patch_orig).reshape(-1))
        print("PGD step {}: score={}, matches={}, norm={}".format(
            i, curr_score, curr_matches, norm)
        )

        if curr_matches > best_matches:
            best_matches = curr_matches
            if best_matches % 10 == 0 or best_matches > 40:
                cv2.imwrite(path.format("_" + str(best_matches)),
                            patch_adv.astype(np.uint8))

        if curr_matches >= 50:
            return patch_adv.astype(np.uint8)

        i += 1

        g = grad_estimate(SIFT, patch_adv, curr_score, n=100, s=7, fp=True)
        patch_adv[:, :, :3] += lr * g[:, :, :3]
        patch_adv = np.clip(patch_adv, 0, 255)

        if i % 10 == 0 and num_rewinds == 0:
            score, num_match = SIFT.score(patch_adv.astype(np.uint8), fp=True)
            print(score, num_match)
            cv2.imwrite(path.format("_curr"), patch_adv.astype(np.uint8))

    return patch_adv.astype(np.uint8)


def postprocess(SIFT, patch, patch_orig, orig_matches):

    assert(patch.shape == patch_orig.shape)

    norm = np.linalg.norm(
        (patch.astype(np.float32) - patch_orig.astype(np.float32)).reshape(-1)
    )
    print("norm before: {}".format(norm))

    h, w, _ = patch.shape
    for i in range(h):
        for j in range(w):
            old = patch[i, j, :].copy()
            patch[i, j, :] = patch_orig[i, j, :].copy()
            score, matches = SIFT.score(patch_adv.astype(np.uint8), fp=True)

            if matches < orig_matches:
                patch[i, j, :] = old.copy()

    norm = np.linalg.norm(
        (patch.astype(np.float32) - patch_orig.astype(np.float32)).reshape(-1))
    print("norm after: {}".format(norm))

    return patch


if __name__ == '__main__':
    np.random.seed(1)

    cv2_sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()

    src_logo_file = sys.argv[1]
    output_dir = sys.argv[2]
    safe_mkdir(output_dir)

    src_logo = cv2.imread(src_logo_file, -1)

    match_threshold = 0.05
    match_threshold_small = 1.0
    SIFT = SIFTModel(src_logo_file, match_threshold=match_threshold,
                     match_threshold_small=match_threshold_small)

    all_white = 255 * np.ones_like(src_logo).astype(np.float32)
    fp = all_white + 40 * np.random.randn(*src_logo.shape)
    fp[:, :, -1] = 20
    fp = np.clip(fp, 0, 255).astype(np.uint8)

    score, num_match = SIFT.score(fp, fp=True)
    patch_adv = PGD(SIFT, fp, path=output_dir + "/fp{}.png")

    score, num_match = SIFT.score(patch_adv, fp=True)
    print(score, num_match)
    cv2.imwrite(output_dir + "/fp.png", patch_adv)

    print('postprocess...')
    patch_adv = postprocess(SIFT, patch_adv, src_logo, num_match)

    score, num_match = SIFT.score(patch_adv, fp=True)
    print(score, num_match)

    cv2.imwrite(output_dir + "/fp_post.png", patch_adv)
