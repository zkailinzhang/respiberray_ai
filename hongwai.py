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
from ctypes import *
from matplotlib import pyplot as plt
from scipy import ndimage, misc
import subprocess
import cv2
import scipy
import traceback


frame=(c_float*768)()

mlx_shape = (24,32) # mlx90640 shape
mlx_interp_val = 10 # interpolate # on each dimension
VIS_bool = True
#VIS_bool = False
mlx90640 = cdll.LoadLibrary('./libmlx90640.so')
def base(cmd):
    if subprocess.call(cmd, shell=True):
        raise Exception("{} failed!".format(cmd))
fig = plt.figure(figsize=(8,6)) # start figure
ax = fig.add_subplot(111) # add subplot
fig.subplots_adjust(0.1,0.05,0.95,0.95) # get rid of unnecessary padding
if VIS_bool:
    
    mlx_interp_shape = (mlx_shape[0]*mlx_interp_val,
                        mlx_shape[1]*mlx_interp_val) # new shape

    therm1 = ax.imshow(np.zeros(mlx_interp_shape),interpolation='none',
                    cmap=plt.cm.bwr,vmin=25,vmax=45) # preemptive image

    cbar = fig.colorbar(therm1) # setup colorbar
    cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) # colorbar label

    fig.canvas.draw() # draw figure to copy background
    ax_background = fig.canvas.copy_from_bbox(ax.bbox) # copy background
    # fig.show() # show the figure before blitting


def DetectWenduRun():
    try:
        # mlx90640 will output 32*24 temperature array with chess mode
        if VIS_bool:
            fig.canvas.restore_region(ax_background) # restore background
        
        mlx90640.get_mlxFrame(frame) # read mlx90640
        
        # print("frame=%.2f"%(np.max(np.array(frame))))
        data_array = np.fliplr(np.reshape(frame,mlx_shape)) # reshape, flip data
        data_array = ndimage.zoom(data_array,mlx_interp_val) # interpolate
        temp_max = np.max(data_array)
        temp_min = np.min(data_array)
        # print("maxTemp=%.2f"%(np.max(data_array)))
        #test save
        
        
        misc.toimage(data_array).save("detect_temperature1.jpg")
        img = cv2.imread("detect_temperature1.jpg")
        text = f'Tmin={temp_min:+.1f}C   Tmax={temp_max:+.1f}C '     
        cv2.putText(img, text, (30, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
        cv2.imwrite("detect_temperature2.jpg",img)
       
        if VIS_bool:
            therm1.set_array(data_array) # set data,更新图像数据而不生成新图
            
            therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
            cbar.on_mappable_changed(therm1) # update colorbar range
            ax.draw_artist(therm1) # draw new thermal image  
            # fig.canvas.blit(ax.bbox) # draw background
            # fig.canvas.flush_events() # show the new image
            plt.savefig('detect_temperature3.jpg',dpi=40)
            if np.max(data_array) > 45.0:
                plt.savefig('detect_temperature4.jpg',dpi=40)
            
        
        filei = open("detect_temperature3.jpg",'rb')
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
    except:
        traceback.print_exc()
        
def DetectWendu(loop):
    import time
    while(1):
        DetectWenduRun()
        
        time.sleep(loop) 


if __name__ == "__main__":
    import time
    while(1):
        DetectWenduRun()
        time.sleep(1)

