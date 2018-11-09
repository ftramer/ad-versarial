import sys
import numpy as np
import cv2
from utils import has_alpha, blend_white

file1 = sys.argv[1]
img1 = cv2.imread(file1, -1).astype(np.float32)

if has_alpha(img1):
    img1 = blend_white(img1).astype(np.float32)

img1 = img1.astype(np.float32)
img1 /= 255.0
img1 = img1*0 #+ 1

print(img1.shape)
norm1 = np.linalg.norm(img1.reshape(-1))
print("norm1: {}".format(norm1))

file2 = sys.argv[2]
img2 = cv2.imread(file2, -1).astype(np.float32)

if has_alpha(img2):
    img2 = blend_white(img2).astype(np.float32)

img2 = img2.astype(np.float32)
img2 /= 255.0

print(img2.shape)
norm2 = np.linalg.norm(img2.reshape(-1))
print("norm2: {}".format(norm2))

diff = np.linalg.norm((img1 - img2).reshape(-1))
print("{:.1f}".format(diff))