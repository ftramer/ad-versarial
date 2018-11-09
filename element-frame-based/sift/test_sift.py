from matplotlib import pyplot as plt
import sys
import cv2
from sift.model import get_keypoints, sift_pred

src_logo_file = sys.argv[1]
target_logo_file = sys.argv[2]
src_logo = cv2.imread(src_logo_file, -1)
target_logo = cv2.imread(target_logo_file, -1)

cv2_sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

src_kp, src_des = get_keypoints(cv2_sift, src_logo)

score, num_matches, matches = \
    sift_pred(cv2_sift, bf, src_kp, src_des, target_logo,
              template_img=src_logo, draw_matches=True, ratio=0.6, fp=False)

print("matches: {}".format(num_matches))
plt.imshow(cv2.cvtColor(matches, cv2.COLOR_BGR2RGB))
plt.show()
