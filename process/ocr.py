from paddleocr import paddleocr, PaddleOCR
import pandas as pd
from glob import glob
import os
import cv2
from tqdm import tqdm
import logging

# 屏蔽调试错误
paddleocr.logging.disable(logging.DEBUG)

class OCR():
    def __init__(self):
        self.ocr = PaddleOCR()

    def scan(self, file_path, output_path, marked_path=None):
        # 文字识别
        info = self.ocr.ocr(file_path, cls=False)
        print(info)
        exit()
        df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'text'])
        for i, item in enumerate(info):
            # 保留左上和右下坐标
            ((x1, y1), _, (x2, y2), _), (text, _) = item
            df.loc[i] = list(map(int, [x1, y1, x2, y2])) + [text]
        # 保存识别结果
        df.to_csv(output_path)
        # 判断是否需要保存标记文件
        if marked_path:
            self.marked(df, file_path, marked_path)

if __name__ == '__main__':
    ocr = OCR()
    ocr.scan('../input/imgs/predict/20190827_163606.jpg', None)