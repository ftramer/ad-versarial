import numpy as np
import io
import tensorflow as tf
import math


def get_size_mul(h, w, target_height):
    if target_height == 0:
        return 1

    mul = target_height / (1.0 * h)
    return int(math.ceil(mul))


def new_dims(h, w, target_height):
    mul = get_size_mul(h, w, target_height)
    return int(mul * h), int(mul * w)


def read_all_chars():
    char_map = {' ': 0}
    with io.open('OCR/tf_tesseract/eng.char', encoding='utf8') as f:
        for line in f:
            arr = line[:-1].split('\t')
            idx = arr[0]
            try:
                idx = int(idx)
            except ValueError:
                idx = idx.split(',')
                idx = [int(i) for i in idx]
            c = arr[1]
            char_map[c] = idx
    return char_map


def dense_to_sparse(label, char_map=None):
    indices = []
    values = []
    if isinstance(label, int):
        label = [label]

    shapes = np.asarray((1, len(label)), dtype=np.int64)
    for i, l in enumerate(label):
        indices.append([0, i])
        if char_map is not None:
            if l == ' ':
                l = 0
            else:
                l = char_map[l]
        values.append(l)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)

    return indices, values, shapes


def normalize(image):
    image = np.clip((image + 1) * 127.5, 0, 255)
    return image.astype(np.uint8)


def remove_alpha(img_ph):
    alpha = img_ph[:, :, 3:]
    rgb = img_ph[:, :, :3]

    return (1 - alpha/255.) * 255. + alpha/255. * rgb


def preprocess_tf(img_ph, h, w, ch=3):
    black, white = 0., 255.
    contrast = (white - black) / 2.
    img_ph = (img_ph - black) / contrast - 1
    img_ph = tf.expand_dims(img_ph, 0)
    return img_ph


def tess_preprocess(img, h, w, d=1):
    img = img.astype(float)
    if d == 3:
        black, white = 0., 255.
        contrast = (white - black) / 2.
        img = (img - black) / contrast - 1
        img = img.reshape(1, h, w, 3)
        return img.astype(np.float32)

    img = img.reshape(h, w)
    y = h / 2
    mid_line = img[y]
    black, white = compute_bw(mid_line)
    contrast = (white - black) / 2.

    if contrast <= 0.:
        contrast = 1.

    img = (img - black) / contrast - 1
    img = img.reshape(1, h, w, 1)
    return img.astype(np.float32)


def ile(arr, q):
    assert 0 <= q <= 1
    bucket = np.zeros(256)

    for a in arr:
        bucket[int(a)] += 1

    total = len(arr)
    target = q * total
    s = 0
    i = 0
    while i < 256 and s < target:
        s += bucket[i]
        i += 1

    if i == 0:
        return 0

    assert bucket[i - 1] > 0
    return i - (s - target) / bucket[i - 1]


def compute_bw(mid_line):
    mins = []
    maxs = []
    prev = mid_line[0]
    curr = mid_line[1]
    for i in range(1, len(mid_line) - 1):
        next = mid_line[i + 1]
        if (curr < prev and curr <= next) or (curr <= prev and curr < next):
            mins.append(curr)
        if (curr > prev and curr >= next) or (curr >= prev and curr > next):
            maxs.append(curr)
        prev = curr
        curr = next

    if len(mins) == 0:
        mins.append(0.0)
    if len(maxs) == 0:
        maxs.append(255.0)

    black = ile(mins, 0.25)
    white = ile(maxs, 0.75)
    return black, white


def decode(sparse_output, char_map, sparse=True):
    inverted_map = {v: k for (k, v) in char_map.items()}
    inverted_map[110] = ' '

    if sparse:
        strings = [[inverted_map[i] for i in out.values] for out in sparse_output]
    else:
        strings = [[inverted_map[i] for i in out] for out in sparse_output]

    strings = [''.join(s) for s in strings]
    return strings


def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])
