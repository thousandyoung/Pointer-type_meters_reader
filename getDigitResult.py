# coding: utf8
import requests
import json
import cv2
import base64
import numpy as np
from collections import Counter
from reinforceDigit import  reinforceDigit
from baiduOCR import Get_baiduOCR_Response
import re

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')

# 输出[{},{}...]


def getOcrResult(r):
    r_list = r.json()['words_result']
    # print(isinstance(r_list,list))
    # sort by 'text'
    # print(r_list)
    r_list = list(
        filter(lambda dict: dict['words'].isdigit(), r_list))               # 过滤非数字
    for item in r_list:
        # 将字符串转为数字
        item['words'] = int(item['words'])
        if item['words'] < 0:                                               # 过滤负数
            r_list.remove(item)
    word_counter = Counter([item['words'] for item in r_list])

    r_list = list(
        filter(lambda dict: word_counter[dict['words']] == 1, r_list))      # 过滤被识别出两次的数字

    r_list.sort(key=lambda dict: int(dict['words']))

    delta_list = []
    for i in range(len(r_list)-1):
        delta_list.append(abs(r_list[i+1]['words']-r_list[i]['words']))
    delta_counter = Counter(delta_list)
    most_common_delta = delta_counter.most_common(1)[0][0]
    if most_common_delta % 5 == 0:
        r_list = list(
            filter(lambda dict: dict['words'] % 5 == 0, r_list))            # 过滤不正常的数字

    return r_list

def getOcrResultForDigitMeter(r):
    r_list = r.json()['words_result']
    # print(isinstance(r_list,list))
    # sort by 'text'
    # print(r_list)
    r_list = list(
        filter(lambda dict: dict['words'].isdigit(), r_list))               # 过滤非数字
    for item in r_list:
        # 将字符串转为数字
        item['words'] = int(item['words'])
        # if item['words'] < 0:                                               # 过滤负数
        #     r_list.remove(item)
    word_counter = Counter([item['words'] for item in r_list])

    # r_list = list(
    #     filter(lambda dict: word_counter[dict['words']] == 1, r_list))      # 过滤被识别出两次的数字

    r_list.sort(key=lambda dict: int(dict['words']))

    # delta_list = []
    # for i in range(len(r_list)-1):
    #     delta_list.append(abs(r_list[i+1]['words']-r_list[i]['words']))
    # delta_counter = Counter(delta_list)
    # most_common_delta = delta_counter.most_common(1)[0][0]
    # if most_common_delta % 5 == 0:
    #     r_list = list(
    #         filter(lambda dict: dict['words'] % 5 == 0, r_list))            # 过滤不正常的数字

    return r_list

if __name__ == '__main__':
    # 发送HTTP请求
    img_path = reinforceDigit("images/digit_3.jpg")
    r = Get_baiduOCR_Response(img_path)
    result = getOcrResultForDigitMeter(r)
    list = r.json()["words_result"]
    for item in list:
        if bool(re.search(r'\d', item["words"])):
            print(item["words"])