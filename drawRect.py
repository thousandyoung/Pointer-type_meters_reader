import cv2
import json

'''
使用pointer.py将ocr识别结果保存到ocrResult.json后
可以画出各个数字的矩形框
'''

with open('ocrResult.json', 'r') as f:
    r = json.load(f)
    r_list = r['words_result']
    r_list = filter(lambda dict: dict['words'].isdigit(), r_list)
    r_list = list(r_list)
    r_list.sort(key=lambda dict: int(dict['words']))


img = cv2.imread('images/7.png')

for word in r_list:
    p = []
    for i in range(4):
        p.append((word['min_finegrained_vertexes_location'][i]['x'],
                  word['min_finegrained_vertexes_location'][i]['y']))
    for i in range(4):
        cv2.line(img, p[i], p[(i+1) % 4], (255, 0, 0))
        
    
cv2.imshow('image', img)
cv2.waitKey()