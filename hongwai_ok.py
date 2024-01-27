# -*- coding: utf-8 -*-
#!/usr/bin/python3

import time
import numpy as np
import datetime 
import cv2
import cmapy
from ctypes import *
from  config import  Config
import pymysql

mlx = cdll.LoadLibrary('./libmlx90640.so')

# function to convert temperatures to pixels on image
def temps_to_rescaled_uints(f,Tmin,Tmax):
    norm = np.uint8((f - Tmin)*255/(Tmax-Tmin))
    norm.shape = (24,32)
    return norm


def DetectWendu():
    width=320
    height=240
    width=240
    height=320
    
    t0 = time.time()
    image=(c_float*768)()

    mlx.get_mlxFrame(image)        
    temp_min,temp_max = np.min(image),np.max(image)
    img= temps_to_rescaled_uints(image,temp_min,temp_max)
    img = cv2.applyColorMap(img, cmapy.cmap('bwr'))
    img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
    img = cv2.resize(img, (height,width), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (height,width), interpolation=cv2.INTER_NEAREST)
    img = cv2.flip(img, 1)

    text = f'Tmin={temp_min:+.1f}C   Tmax={temp_max:+.1f}C   FPS={1/(time.time() - t0):.1f} '     
    cv2.putText(img, text, (30, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
    cv2.imwrite("alert.jpg",img)
    
    filei = open("alert.jpg",'rb')
    dataimg = filei.read()

    time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

    db = pymysql.connect(host=Config.mysql_host, port=Config.mysql_port, user=Config.mysql_user, \
        password=Config.mysql_password, database=Config.mysql_database,charset='utf8mb4')
    cursor = db.cursor()

    sql_bj = "INSERT INTO diagnosis_history(task_name,fault_id,data_value,image,time_start) values(%s,%s,%s,%s,%s);"
    cursor.execute(sql_bj,("温度检测",0,temp_max,dataimg,time_now))
    db.commit()

    if temp_max > 45.0: 
        f = open('wendubaojing.txt','w')
        min = '%.2f' %temp_min
        max = '%.2f' %temp_max
        string1 = "temp_min = " + min , "temp_max = " + max
        f.write(str(string1))
        f.close()
        time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')
        sql_bj = "INSERT INTO diagnosis_list(task_name,fault_id,data_value,image,time_start) values(%s,%s,%s,%s,%s);"
        cursor.execute(sql_bj,("温度检测",1,temp_max,dataimg,time_now))
        db.commit()

    del dataimg
    db.close()
    


if __name__ == "__main__":

    Detect()

