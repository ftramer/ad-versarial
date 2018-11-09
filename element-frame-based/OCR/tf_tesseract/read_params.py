import numpy as np


BASE_PATH = 'OCR/tf_tesseract/params/'
CNN_PATH = BASE_PATH + 'fc_params2.txt'
CTC_PATH = BASE_PATH + 'fc_params8.txt'

LFYS64_PATH = BASE_PATH + 'lstm_params4.txt'
LFX128_PATH = BASE_PATH + 'lstm_params5.txt'
LRX128_PATH = BASE_PATH + 'lstm_params6.txt'
LFX256_PATH = BASE_PATH + 'lstm_params7.txt'


def read_params(p):
    print("Reading from {}".format(p))
    with open(p) as f:
        lines = f.readlines()
        weights = []
        for l in lines[1:-2]:
            weights.append(l.split(' ')[1:])
        weights = np.asarray(weights, dtype=float)
        bias = np.asarray(lines[-1].split(' ')[1:], dtype=float)
    return [weights, bias]


def read_lstm_params(p):
    print("Reading from {}".format(p))
    wis = []
    wos = []
    bs = []
    with open(p) as f:
        lines = f.readlines()
        i_idx = []
        o_idx = []
        b_idx = []
        for i, l in enumerate(lines):
            if 'inputs' in l:
                i_idx.append(i)
            if 'outputs' in l:
                o_idx.append(i)
            if 'bias' in l:
                b_idx.append(i)

        for i, j, k in zip(i_idx, o_idx, b_idx):
            wi = []
            for l in lines[i + 1: j]:
                wi.append(l.split(' ')[1:])
            wo = []
            for l in lines[j + 1: k]:
                wo.append(l.split(' ')[1:])
            wi = np.asarray(wi, dtype=float)
            wo = np.asarray(wo, dtype=float)
            b = np.asarray(lines[k + 1].split(' ')[1:], dtype=float)
            wis.append(wi)
            wos.append(wo)
            bs.append(b)
    return wis, wos, bs


def reshape_lstm_params(params, use_gpu=True):
    wis, wos, bs = params
    wi = np.hstack((wis[1], wis[0], wis[2], wis[3]))
    wo = np.hstack((wos[1], wos[0], wos[2], wos[3]))
    b = np.concatenate((bs[1], bs[0], bs[2], bs[3]))
    # print wi.shape, wo.shape
    if use_gpu:
        return [np.vstack([wi, wo]), b]
    else:
        return [wi, wo.reshape(-1, 4, wos[0].shape[0]), b]


def read_tesseract_params(use_gpu=True):
    params = []
    cnn_params = read_params(CNN_PATH)
    cnn_params[0] = cnn_params[0].reshape(3, 3, 1, 16).transpose(1, 0, 2, 3)
    params += cnn_params
    params += reshape_lstm_params(read_lstm_params(LFYS64_PATH), use_gpu=use_gpu)
    params += reshape_lstm_params(read_lstm_params(LFX128_PATH), use_gpu=use_gpu)
    params += reshape_lstm_params(read_lstm_params(LRX128_PATH), use_gpu=use_gpu)
    params += reshape_lstm_params(read_lstm_params(LFX256_PATH), use_gpu=use_gpu)
    params += read_params(CTC_PATH)
    return params


if __name__ == '__main__':
    read_tesseract_params()