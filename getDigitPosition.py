# coding: utf8
from numpy.core.fromnumeric import mean
import requests
import json
import cv2
import base64
import numpy as np
from sympy.simplify.fu import RL1
from getDigitResult import getOcrResult


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


# 返回((x0,y0), (x1,y1))
# 1为0之后的第一个示数
# r = requests.post(url=url, headers=headers, data=json.dumps(data))


def get_p0_p1(r):

    r_list = getOcrResult(r)

    text_box_position_0 = r_list[0]['min_finegrained_vertexes_location']
    text_box_position_1 = r_list[1]['min_finegrained_vertexes_location']
    position_0 = get_text_box_position(text_box_position_0)
    position_1 = get_text_box_position(text_box_position_1)
    text_0 = r_list[0]['words']
    text_1 = r_list[1]['words']
    # print(position_0, position_1)

    prob = np.mean([dict['probability']['average'] for dict in r_list])

    return position_0, position_1, text_0, text_1, prob


# 返回[x,y]
def get_text_box_position(l):
    x_sum, y_sum = 0, 0
    for item in l:
        x_sum += item['x']
        y_sum += item['y']
    return int(x_sum / 4), int(y_sum / 4)


if __name__ == '__main__':
    # 发送HTTP请求
    data = {'images': [cv2_to_base64(cv2.imread("5.jpg"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/chinese_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    # 返回结果
    print(get_0_1_position(r))
    get_0_1_position(r)
