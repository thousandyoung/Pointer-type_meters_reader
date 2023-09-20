import requests
import base64
import json
import re
'''
通用文字识别（高精度含位置版）
'''


def Get_baiduOCR_Response(inputpath):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    # 二进制方式打开图片文件
    f = open(inputpath, 'rb')
    img = base64.b64encode(f.read())

    host = "your API"
    response = requests.get(host)

    params = {"image": img, 'vertexes_location': 'true'}
    access_token = response.json()['access_token']
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        with open('ocrResult.json', 'w') as f:
            json.dump(response.json(), f, sort_keys=True, indent=4)
        # print(response.json())
        return response
