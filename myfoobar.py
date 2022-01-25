import numpy as np
import tqdm

def int_img(original_img, imglength):
    # first line and first column
    for x in range(1, imglength):
        original_img[0][x] += original_img[0][x - 1]
        original_img[x][0] += original_img[x - 1][0]
    # rest of the image
    for i in range(1, imglength):
        for j in range(1, imglength):
            original_img[i][j] += (original_img[i - 1][j] + original_img[i][j - 1] - original_img[i - 1][j - 1])

def Harr1(integral_img, loc):
    i, j, length, width = loc[1], loc[2], loc[3] - loc[1] + 1, loc[4] - loc[2] + 1
    white = integral_img[i + length - 1][j - 1 + width // 2] + integral_img[i - 1][j - 1] \
            - integral_img[i + length - 1][j - 1] - integral_img[i - 1][j - 1 + width // 2]
    black = integral_img[i + length - 1][j - 1 + width] + integral_img[i - 1][j - 1 + width // 2] \
            - integral_img[i + length - 1][j - 1 + width // 2] - integral_img[i - 1][j - 1 + width]
    diff = white - black
    return diff

def Harr2(integral_img, loc):
    i, j, length, width = loc[1], loc[2], loc[3] - loc[1] + 1, loc[4] - loc[2] + 1
    white = integral_img[i + length - 1][j - 1 + width] + integral_img[i - 1 + length // 2][j - 1] \
            - integral_img[i - 1 + length][j - 1] - integral_img[i - 1 + length // 2][j - 1 + width]
    black = integral_img[i - 1 + length // 2][j - 1 + width] + integral_img[i - 1][j - 1] \
            - integral_img[i - 1 + length // 2][j - 1] - integral_img[i - 1][j - 1 + width]
    diff = white - black
    return diff

def Harr3(integral_img, loc):
    i, j, length, width = loc[1], loc[2], loc[3] - loc[1] + 1, loc[4] - loc[2] + 1
    whiteleft = integral_img[i + length - 1][j - 1 + width//3] + integral_img[i - 1][j - 1] \
            - integral_img[i + length - 1][j - 1] - integral_img[i - 1][j - 1 + width//3]
    whiteright = integral_img[i + length - 1][j - 1 + width] + integral_img[i - 1][j - 1 + 2*width//3] \
            - integral_img[i - 1 + length][j - 1 + 2*width//3] - integral_img[i - 1][j - 1 + width]
    blackmiddle = integral_img[i - 1 + length][j - 1 + 2*width//3] + integral_img[i - 1][j - 1 + width//3] \
            - integral_img[i - 1 + length][j - 1 + width//3] - integral_img[i - 1][j - 1 + 2*width//3]
    diff = whiteleft + whiteright - blackmiddle
    return diff

def Harr4(integral_img, loc):
    i, j, length, width = loc[1], loc[2], loc[3] - loc[1] + 1, loc[4] - loc[2] + 1
    white_ll = integral_img[i + length - 1][j - 1 + width//2] + integral_img[i - 1 + length//2][j - 1] \
            - integral_img[i + length - 1][j - 1] - integral_img[i - 1 + length//2][j - 1 + width//2]
    white_ur = integral_img[i - 1 + length//2][j - 1 + width] + integral_img[i - 1][j - 1 + width//2] \
            - integral_img[i - 1 + length//2][j - 1 + width//2] - integral_img[i - 1][j - 1 + width]
    black_ul = integral_img[i - 1 + length//2][j - 1 + width//2] + integral_img[i - 1][j - 1] \
            - integral_img[i + length - 1][j - 1] - integral_img[i - 1][j - 1 + width//2]
    black_lr = integral_img[i - 1 + length][j - 1 + width] + integral_img[i - 1 + length//2][j - 1 + width//2] \
            - integral_img[i + length - 1][j - 1 + width//2] - integral_img[i - 1 + length//2][j - 1 + width]
    diff = white_ll + white_ur - black_lr - black_ul
    return diff

def ERMdecisionstumps(trainarray, D):
    rows, cols = len(trainarray), len(trainarray[0])
    Fstar = np.inf
    for j in tqdm.trange(cols-1):
        indices = trainarray[:, j].argsort()        # ascending indices according to jth column
        D_sorted = D[indices]                       # Distribution sorted
        trainarray_sorted = trainarray[indices]     # trainarray sorted
        yi_eq_1_idx = np.where(trainarray_sorted[:, -1] == 1)
        F = sum(D_sorted[yi_eq_1_idx])
        if F < Fstar:
            Fstar, thetastar, jstar = F, trainarray_sorted[0][j] - 1, j
        for i in range(rows):
            F = F - trainarray_sorted[i][-1] * D_sorted[i]
            if F < Fstar and trainarray_sorted[i][j] != trainarray_sorted[i+1][j]:
                Fstar, thetastar, jstar = F, (trainarray_sorted[i][j] + trainarray_sorted[i+1][j])/2, j
    return jstar, thetastar # jstar is the actual column index, use directly

def testERM(testarray, jstar, thetastar):
    x = testarray[:, jstar]
    label = np.sign(thetastar - x)
    return label

def testingERM(feature_value, thetastar):
    label = np.sign(thetastar - feature_value)
    # label = thetastar - feature_value
    return label

