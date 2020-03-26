# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import cv2 as cv
import math
import os
from time import time
from skimage import data


def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    min, max = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    # digitize
    bins = np.linspace(min, max + 1, nbit + 1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:, 1:], gl1[:, -1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    # glcm.shape = (8, 8, 638, 1201)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i, j] = cv.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm mean
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    std2 = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i, j] * i - mean) ** 2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i, j] * (i - j) ** 2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i, j] * np.abs(i - j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i, j] / (1. + (i - j) ** 2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h, w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm += glcm[i, j] ** 2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_ = np.max(glcm, axis=(0, 1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
    return ent


def max_gray_levels(img):
    max_gray_level = 0
    (height, width) = img.shape
    #     print (height,width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def get_glcm(img, d_x, d_y, gray_level=8):
    srcdata = img.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = img.shape

    max_gray_level = max_gray_levels(img)

    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def get_glcm_feature_imgs(tiles_dir, save=True, verbose=False):
    tiles_dir = os.path.abspath(tiles_dir)
    tiles = os.listdir(tiles_dir)
    tiles_num = len(tiles)

    for i in range(tiles_num):
        # print("Extract feature of image {}".format(tiles[i]))
        img_name = tiles_dir + "/" + tiles[i]
        if tiles[i][-3:] == "npy":
            img = np.load(img_name)
            if len(img.shape) > 2:
                if img.shape[2] > 4:
                    img = img[:, :, :4]
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                # plt.figure()
                # plt.imshow(img)
        else:
            img = cv.imread(img_name)
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        homogeneity = fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5)
        if save:
            # feature_dir = os.path.abspath(os.path.join(tiles_dir,os.path.pardir)) + "/glcm_images/"
            feature_dir = os.path.join(tiles_dir, os.path.pardir) + "/glcm_images/"
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            np.save(os.path.join(feature_dir, '{}'.format(tiles[i])), homogeneity)
        if verbose:
            print("正在处理: {}".format(tiles[i]))
            plt.figure()
            plt.imshow(homogeneity)
    return homogeneity


def feature_computer(p, gray_level=8):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])

    return Asm, Con, -Eng, Idm


def fetch_features(tiles_dir, verbose=False):  # 特征提取

    t0 = time()
    tiles_dir = os.path.abspath(tiles_dir)
    tiles = os.listdir(tiles_dir)
    tiles.sort(key=lambda x: int(x[:-8]), reverse=False)  # 根据 Xtile.npy 的X排序, reverse=False 升序
    tiles_num = len(tiles)
    feature_total = []

    for i in range(tiles_num):
        print("Extract feature of image {}".format(tiles[i]))
        img_name = tiles_dir + "/" + tiles[i]
        if tiles[i][-3:] == "npy":
            img = np.load(img_name)
            if len(img.shape) > 2:
                if img.shape[2] > 4:
                    img = img[:, :, :4]
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                # plt.figure()
                # plt.imshow(img)
        else:
            img = cv.imread(img_name)
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        img_average = np.mean(img)
        img_std = np.std(img)
        img_values = [img_average, img_std]

        feature_glcm = []
        # feature_glcm.append(tiles[i])
        # feature_glcm.extend(img_values)

        ##Gabor

        ##小波

        ##灰度共生
        glcm_0 = get_glcm(img, 1, 0)

        g_v = feature_computer(glcm_0)
        # feature_glcm.append(g_v[1])
        # feature_glcm.append(g_v[2]) Con, -Eng, Idm
        # feature_glcm.append(g_v[3])
        feature_glcm.append({"tile_name": int(tiles[i][:-8]), "img_mean": img_average, "img_std": img_std, \
                             "Con": g_v[1], "Eng": g_v[2], "Idm": g_v[3], })
        # print("{}已得到灰度共生矩阵！".format(tiles[i]))

        feature_total.extend(feature_glcm)
        # print("{}的特征已得到！".format(tiles[i]))

    print("特征获取结束")
    t1 = time()
    print('Extract {} tiles: {:0.3f}s'.format(tiles_num, t1 - t0))
    return tiles_num, feature_total

# 在圆形选取框基础上，加入旋转不变操作
def rotation_invariant_LBP(img, radius=3, neighbors=8):
    h, w=img.shape
    dst = np.zeros((h-2*radius, w-2*radius),dtype=img.dtype)
    for i in range(radius,h-radius):
        for j in range(radius,w-radius):
            # 获得中心像素点的灰度值
            center = img[i,j]
            for k in range(neighbors):
                # 计算采样点对于中心点坐标的偏移量rx，ry
                rx = radius * np.cos(2.0 * np.pi * k / neighbors)
                ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
                # 为双线性插值做准备
                # 对采样点偏移量分别进行上下取整
                x1 = int(np.floor(rx))
                x2 = int(np.ceil(rx))
                y1 = int(np.floor(ry))
                y2 = int(np.ceil(ry))
                # 将坐标偏移量映射到0-1之间
                tx = rx - x1
                ty = ry - y1
                # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
                w1 = (1-tx) * (1-ty)
                w2 =    tx  * (1-ty)
                w3 = (1-tx) *    ty
                w4 =    tx  *    ty
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i+y1,j+x1] * w1 + img[i+y2,j+x1] *w2 + img[i+y1,j+x2] *  w3 +img[i+y2,j+x2] *w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i-radius,j-radius] |= (neighbor>center)  <<  (np.uint8)(neighbors-k-1)
    # 进行旋转不变处理
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            currentValue = dst[i,j]
            minValue = currentValue
            for k in range(1, neighbors):
                # 对二进制编码进行循环左移，意思即选取移动过程中二进制码最小的那个作为最终值
                temp = (np.uint8)(currentValue>>(neighbors-k)) |  (np.uint8)(currentValue<<k)
                if temp < minValue:
                    minValue = temp
            dst[i,j] = minValue

    return dst
