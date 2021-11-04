import datetime
import pandas as pd
from random import sample
import cv2
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from sympy import *
import math
from getDigitPosition import get_p0_p1, cv2_to_base64
from getDigitResult import getOcrResult
import requests
import json
from baiduOCR import Get_baiduOCR_Response
import math
import argparse
from re import T
from sympy.logic.inference import PropKB
import os
import shutil


class mential():
    def get_max_point(self, cnt):
        lmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        tmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        pmost = [lmost, rmost, tmost, bmost]
        return pmost

    def distance(self, pmost, centerpoint):
        cx, cy = centerpoint
        distantion = []
        for point in pmost:
            dx, dy = point
            distantion.append((cx - dx) ** 2 + (cy - dy) ** 2)
        index_of_max = distantion.index((max(distantion)))
        return index_of_max

    def ds_ofpoint(self, a, b):
        x1, y1 = a
        x2, y2 = b
        distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        return distances

    def findline(self, cp, lines):
        x, y = cp
        cntareas = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            aa = sqrt(min((x1 - x) ** 2 + (y1 - x) **
                          2, (x2 - x) ** 2 + (y2 - x) ** 2))
            if (aa < 50):
                cntareas.append(line)
        print(cntareas)
        return cntareas


def angle(v1, v2):
    vec1 = (v2[2]-v2[0], v2[3]-v2[1])    # 圆心到识别点的向量
    vec2 = (v1[2]-v1[0], v1[3]-v1[1])    # 圆心到指针端点的向量
    Norm = np.linalg.norm(vec1)*np.linalg.norm(vec2)            # |a||b|
    # 叉乘计算sinα，判断顺逆时针
    alpha = np.arcsin(np.cross(vec1, vec2)/Norm)
    theta = np.rad2deg(np.arccos(np.dot(vec1, vec2)/Norm))      # 点乘计算cosθ，得到夹角
    if alpha < 0:       # vec1和vec2成左手系，theta即为vec1到vec2顺时针方向夹角
        return theta
    else:               # vec1和vec2成右手系，360-theta即为vec1到vec2顺时针方向夹角
        return 360-theta


def get_mode(arr):
    while 0 in arr:
        arr.remove(0)
    mode = []
    arr_appear = dict((a, arr.count(a)) for a in arr)  # 统计各个元素出现的次数
    if max(arr_appear.values()) == 1:  # 如果最大的出现为1
        arrs = np.array(arr)
        oo = np.median(arrs)
        return oo
    else:
        for k, v in arr_appear.items():  # 否则，出现次数最大的数字，就是众数
            if v == max(arr_appear.values()):
                mode.append(k)
    return mode

#
# def remove_diff(deg):
#     """
#     :funtion :
#     :param b:
#     :param c:
#     :return:
#     """
#     if (True):
#         # new_nums = list(set(deg)) #剔除重复元素
#         mean = np.mean(deg)
#         var = np.var(deg)
#         # print("原始数据共", len(deg), "个\n", deg)
#         '''
#         for i in range(len(deg)):
#             print(deg[i],'→',(deg[i] - mean)/var)
#             #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
#         '''
#         # print("中位数:",np.median(deg))
#         percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
#         # print("分位数：", percentile)
#         # 以下为箱线图的五个特征值
#         Q1 = percentile[0]  # 上四分位数
#         Q3 = percentile[2]  # 下四分位数
#         IQR = Q3 - Q1  # 四分位距
#         ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
#         llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值
#
#         new_deg = []
#         uplim = []
#         for i in range(len(deg)):
#             if (llim < deg[i] and deg[i] < ulim):
#                 new_deg.append(deg[i])
#         # print("清洗后数据共", len(new_deg), "个\n", new_deg)
#     new_deg = np.mean(new_deg)
#
#     return new_deg
#     # 图表表达


flag = 0
p0 = 0


def markzero(path):
    img = cv2.imread(path)

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global flag, p0
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            p0 = [x, y]
            print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), thickness=-1)
            cv2.putText(img, '*0*', (x - 30, y), 1,
                        2.0, (0, 0, 0), thickness=2)
            # # cv2.imshow("image", img)

        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键fang
            cv2.destroyWindow("image")
            print(p0)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

    # cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return p0
    # while (1):
    #     # cv2.imshow("image", img)
    #     if cv2.waitKey(0)&0xFF>0:
    #     # if cv2.waitKey(500)|0xFF>0:
    #         print(flag)
    #         break


def cut_pic(path):
    """
    :param pyrMeanShiftFiltering(input, 10, 100) 均值滤波
    :param 霍夫概率圆检测
    :param mask操作提取圆
    :return: 半径，圆心位置
    """
    input = cv2.imread(path)
    dst = cv2.pyrMeanShiftFiltering(input, 10, 100)

    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1,
                               80, param1=100, param2=20, minRadius=80, maxRadius=0)
    circles = np.uint16(np.around(circles))  # 把类型换成整数
    r_1 = circles[0, 0, 2]
    c_x = circles[0, 0, 0]
    c_y = circles[0, 0, 1]
    # print(input.shape[:2])
    circle = np.ones(input.shape, dtype="uint8")
    circle = circle * 255
    # print(circle)
    cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)
    # cv2.circle(circle, (c_x, c_y), int(r_1*0.65), (255,255,255), -1)
    # # cv2.imshow("circle", circle)
    bitwiseOr = cv2.bitwise_or(input, circle)

    cv2.circle(bitwiseOr, (c_x, c_y), 2, 0, -1)
    # # cv2.imshow(pname+'_resize'+ptype, bitwiseOr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(pname + '_resize' + ptype, bitwiseOr)
    ninfo = [r_1, c_x, c_y]
    return ninfo


def linecontours(cp_info, path):
    """
    :funtion : 提取刻度线，指针
    :param a: 高斯滤波 GaussianBlur，自适应二值化adaptiveThreshold，闭运算
    :param b: 轮廓寻找 findContours，
    :return:kb,new_needleset
    """
    r_1, c_x, c_y = cp_info
    img = cv2.imread(path)

    # cv2.imshow('image_raw', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # cv2.imshow('image_gaussblur', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度

    circle = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(circle, (c_x, c_y), int(r_1), 255, -1)
    gray = cv2.bitwise_and(gray, circle)
    # cv2.imshow('image_gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # reinforce rgb 防止其他颜色指针无了
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(imgYUV)
    r = cv2.inRange(V, 170, 255)
    # --- adding up all the results above ---
    img = cv2.add(gray, r)
    # img = gray
    # img = cv2.add(m1, img)
    # img = cv2.add(m2, img)
    # circle = np.zeros(img.shape, dtype="uint8")
    # cv2.circle(circle, (c_x, c_y), int(r_1*0.8), 255, -1)
    # img = cv2.bitwise_and(img, circle)
    # cv2.imshow('image_reinforce', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    binary = cv2.adaptiveThreshold(~img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    circle = np.zeros(binary.shape, dtype="uint8")
    cv2.circle(circle, (c_x, c_y), int(r_1*0.7), 255, -1)
    binary = cv2.bitwise_and(binary, circle)

    # cv2.imshow('image_binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # 开运算 让指针断开
    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(binary, kernel, iterations=2)
    # dilation = cv2.dilate(erosion, kernel, iterations=2)
    #
    # # cv2.imshow('image_binary_after_open', binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 闭运算 加强指针
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=2)
    kernel2 = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(dilation, kernel2, iterations=2)

    # cv2.imshow('image_binary_after_close',erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ************************
    # # cv2.imshow('dds', binary)
    # cv2.waitKey(200)

    contours, hier = cv2.findContours(
        erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(binary, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("contours", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    needlecnt = []  # 指针轮廓集合
    lineCount = 0
    for xx in contours:
        rect = cv2.minAreaRect(xx)
        # print(rect)
        center_of_rect, w_h_of_rect, angle_of_rect = rect
        box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
        # 找指针（找最靠近圆心的点
        point_near_center = box[0]
        dis_between_o_p = (
            point_near_center[0] - c_x)**2 + (point_near_center[1] - c_y)**2
        for point in box:
            dis = (point[0]-c_x)**2 + (point[1]-c_y)**2
            if(dis < dis_between_o_p):
                point_near_center = point
                dis_between_o_p = dis

        w, h = w_h_of_rect
        w = int(w)
        h = int(h)
        ''' 
        从轮廓中筛选出指针
        '''
        if h == 0 or w == 0:
            pass
        else:
            if w > r_1 / 3 or h > r_1 / 3:
                lineCount += 1
                if dis_between_o_p <= (r_1 * 0.3) ** 2:
                    needlecnt.append(xx)  # 指针轮廓

    # 指针穿过圆心的情况
    if len(needlecnt) == 0 and lineCount != 0:
        for xx in contours:
            rect = cv2.minAreaRect(xx)
            # print(rect)
            center_of_rect, w_h_of_rect, angle_of_rect = rect
            w, h = w_h_of_rect
            w = int(w)
            h = int(h)
            ''' 
            从轮廓中筛选出指针
            '''
            if h == 0 or w == 0:
                pass
            else:
                if w > r_1 / 3 or h > r_1 / 3:
                    needlecnt.append(xx)  # 指针轮廓
    ############################################################
    mask = np.zeros(img.shape[0:2], np.uint8)
    print("needlecnt size {}".format(len(needlecnt)))
    # for cnt in needlecnt:
    #     cv2.fillConvexPoly(mask,cnt , 255)
    mask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜
    # # cv2.imshow('needle mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(pname + '_scale' + ptype, img)
    cv2.imwrite(pname + '_needle' + ptype, mask)
    return mask

# img : mask


def needle(path, img, r, cx, cy, x0, y0):
    oimg = cv2.imread(path)
    # circle = np.ones(img.shape, dtype="uint8")
    # circle = circle * 255

    # cv2.imshow('needle mask', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # kernel = np.ones((3, 3), np.uint8)
    # # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀
    #
    # # cv2.imshow('needle after dilate ', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100,
                            minLineLength=int(r / 3), maxLineGap=4)

    nmask = np.zeros(img.shape, np.uint8)

    '''add line to nmask
    record the longest line as pointer'''
    px1, py1, px2, py2 = lines[0][0]
    pointer_len = (py1-py2)**2 + (px1 - px2)**2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        len = (x1 - x2)**2 + (y1-y2)**2
        if len > pointer_len:
            pointer_len = len
            px1, py1, px2, py2 = line[0]
        cv2.line(nmask, (x1, y1), (x2, y2), 100, 1, cv2.LINE_AA)

    # 取最长line 分辨出线的起点和终点
    d1 = (px1 - cx) ** 2 + (py1 - cy) ** 2
    d2 = (px2 - cx) ** 2 + (py2 - cy) ** 2
    if d1 > d2:
        axit = [px1, py1]
    else:
        axit = [px2, py2]
    # nmask = cv2.erode(nmask, kernel, iterations=2)

    # cv2.imshow('houghlines', nmask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cnts, hier = cv2.findContours(
        nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areass = [cv2.contourArea(x) for x in cnts]
    # print(len(areass))
    i = areass.index(max(areass))
    # print('contours[i]',contours[i])
    # cv2.drawContours(img, contours[i], -1, (10,20,250), 1)
    # # cv2.imshow('need_next', img)
    cnt = cnts[i]
    output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
    k = output[1] / output[0]
    k = round(k[0], 2)
    b = output[3] - k * output[2]
    b = round(b[0], 2)
    x1 = cx
    x2 = axit[0]
    # y1 = int(k * x1 + b)
    y1 = cy
    y2 = int(k * x2 + b)
    cv2.line(oimg, (x1, y1), (x2, y2), (0, 23, 255), 1, cv2.LINE_AA)
    cv2.line(oimg, (x1, y1), (x0, y0), (0, 23, 255), 1, cv2.LINE_AA)
    # cv2.imshow('msss', oimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('test_out.jpg')
    cv2.imwrite("test_out.jpg", oimg)
    return x1, y1, x2, y2


pname, ptype = 0, 0
ninfo = None

# angle between opoint and needle


def GetAngle(path, outpath, opoint):
    x0, y0 = opoint
    global pname, ninfo
    pname = outpath + '/' + pname.split('/')[-1]
    print(pname)

    start = datetime.datetime.now()
    mask = linecontours(ninfo, path)  # mask 刻度线的mask

    cx, cy = ninfo[1:]  # 圆心坐标
    da, db, dc, de = needle(path, mask, ninfo[0], cx, cy, x0, y0)  # 圆心和识别点的坐标
    # da,db,dc,de=needle_line(lines,new_needleset,cx,cy)
    # print(da,db,dc,de)

    OZ = [cx, cy, x0, y0]   # 圆心和识别点的坐标
    OP = [cx, cy, dc, de]   # 圆心和指针端点的坐标
    ang1 = angle(OZ, OP)
    output = ang1

    end = datetime.datetime.now()
    print(end - start)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    return output


# 输入图片路径，返回读数

results = []
cur_prob = 0


def GetpointerResult(inputPath):
    # 输入文件夹
    inputpath = inputPath
    # 输出文件夹
    outputpath = 'output'

    r = Get_baiduOCR_Response(inputpath)

    global cur_prob, results
    p0, p1, text0, text1, prob = get_p0_p1(r)
    print("text0 {}, text1 {}".format(text0, text1))

    ang0 = GetAngle(inputpath, outputpath, p0)  # 指针和p0的夹角
    ang1 = GetAngle(inputpath, outputpath, p1)  # 指针和p1的夹角
    ang_between_0_1 = ang0 - ang1  # p0和p1的夹角
    if ang_between_0_1 < 0:
        ang_between_0_1 += 360
    print("ang between first and second text  {} ".format(ang_between_0_1))
    print("ang of first text  {} ".format(ang0))
    print("ang of second text  {} ".format(ang1))

    value = text0 + (text1 - text0) * ang0 / ang_between_0_1  # 计算示数
    print(value)

    print(f'probability: {prob}')
    cur_prob = prob
    results.append({'value': value, 'probability': prob})

    return value


def get_results():
    global results
    if cur_prob > 0.99:
        return results[-1]

    results.sort(key=lambda d: d['value'])
    i = 0
    while True:
        if i == len(results)-1:
            break

        # 将相近的示数合并，对示数取平均，prob取和
        if 0.9 < results[i]['value']/results[i+1]['value'] and results[i]['value']/results[i+1]['value'] < 1.1:
            new_value = (results[i]['value']+results[i+1]['value'])/2
            new_prob = results[i]['probability']+results[i+1]['probability']
            results.remove(results[i])  # remove i
            results.remove(results[i])  # remove i+1
            results.insert(
                i, {'value': new_value, 'probability': new_prob})   # 新的示数
        else:   # 两数相差较大，不能合并，处理下一个数字
            i += 1

    return max(results, key=lambda d: d['probability'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='path of the image')
    parser.add_argument('--detect_direction', type=bool,
                        default=True, help='detect direction or not')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cur_path = args.path
    pname, ptype = args.path.split('.')
    ptype = '.' + ptype

    rotation_degree = [i for i in range(
        30, 361, 30)] if args.detect_direction == True else [0]
    current_degree = 0

    ans = 0

    for r_degree in rotation_degree:
        try:
            ninfo = cut_pic(cur_path)  # 截取表盘
            print('-'*30, f'degree: {current_degree}', '-'*30)
            temp = GetpointerResult(cur_path)

            if cur_prob > 0.99:         # 置信度足够高，直接停，提高效率
                break

            raise Exception('next degree')
        except Exception as e:
            record = False
            print(type(e), ':', e)
            current_degree = r_degree
            img = cv2.imread(args.path)
            (h, w) = img.shape[:2]
            cx, cy = (w//2, h//2)
            M = cv2.getRotationMatrix2D((cx, cy), r_degree, 1)  # 旋转矩阵
            cos_theta = abs(M[0, 0])
            sin_theta = abs(M[1, 0])
            w_new = int(sin_theta*h+cos_theta*w)
            h_new = int(cos_theta*h+sin_theta*w)
            delta_cx, delta_cy = w_new//2-cx, h_new//2-cy
            M[0, 2] += delta_cx
            M[1, 2] += delta_cy
            img = cv2.warpAffine(img, M, (w_new, h_new),
                                 borderValue=(255, 255, 255))
            if not os.path.exists('.tmp'):
                os.mkdir('.tmp',)
            cv2.imwrite(f'.tmp/rotated_{r_degree}'+ptype, img)
            cur_path = f'.tmp/rotated_{r_degree}'+ptype
    if os.path.exists('.tmp'):
        shutil.rmtree('.tmp', ignore_errors=True)

    result = get_results()
    print('-'*30, '示数', '-'*30)
    print(f'value: {result["value"]}')
