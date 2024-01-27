# -*- coding: utf-8 -*-
#!/usr/bin/python3

import time
import numpy as np
import datetime as dt
import cv2
import cmapy
from scipy import ndimage
from ctypes import *

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties

import pandas as pd
import subprocess
import yolo5detect
import hongwai



def imgsConcatShow(scale, imgarray):
    rows = len(imgarray)         
    cols = len(imgarray[0])      
    rowsAvailable = isinstance(imgarray[0], list)

    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)

                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows  
        for x in range(0, rows):
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   
    else:
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver



if __name__ == "__main__":
        
    try:
        cap=cv2.VideoCapture(0)
        #4个这样的分辨率
        width=320
        height=240

        while (cap.isOpened() ):

            ret,raw_frame=cap.read()
            
            #原图resize和红外一样，
            frame = cv2.resize(raw_frame, (height,width), cv2.INTER_AREA)
            
            #热图像
            hongwai_img = hongwai.Detect(raw_frame)
            
            #横向显示，左普通相机右红外
            stackedimageH1 = imgsConcatShow(1, ([frame,hongwai_img]))
            
            #dishui
            names_path = './coco.dishui.names'
            model_path = './yolov5s.dishui.onnx'
            dishui_frame = yolo5detect.Detect(raw_frame,names_path,model_path)
            dishui_img = cv2.resize(dishui_frame, (height,width), cv2.INTER_AREA)

            #louyou
            names_path = './coco.louyou.names'
            model_path = './yolov5s.louyou.onnx'
            louyou_frame = yolo5detect.Detect(raw_frame,names_path,model_path)
            louyou_img = cv2.resize(louyou_frame, (height,width), cv2.INTER_AREA)
            
            #yibiao todo..

            stackedimageH2 = imgsConcatShow(1, ([dishui_img,louyou_img]))
            stackedimage = imgsConcatShow(1, ([stackedimageH1],[stackedimageH2]))
            cv2.imshow("stackedimage",stackedimage)
            cv2.waitKey(50)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Code Stopped by User")

    cv2.destroyAllWindows()
    cap.release()
