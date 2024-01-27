'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
import time
from config import Config


'''
优化：

检测圆：合理的圆半径，结合图像宽高，平均化，圆心和半径。
检测线：线长度超过半径删除
光照，增加亮度

'''

class Reader():
    def __init__(self) -> None:
        pass
        
    def avg_circles(self,circles, b):
        avg_x=0
        avg_y=0
        avg_r=0
        for i in range(b):
            #平均圆心 半径
            avg_x = avg_x + circles[0][i][0]
            avg_y = avg_y + circles[0][i][1]
            avg_r = avg_r + circles[0][i][2]
        avg_x = int(avg_x/(b))
        avg_y = int(avg_y/(b))
        avg_r = int(avg_r/(b))
        return avg_x, avg_y, avg_r

    def dist_2_pts(self,x1, y1, x2, y2):
        #print np.sqrt((x2-x1)^2+(y2-y1)^2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def calibrate_gauge(self,zhizhen,gauge_number,name, file_type):

        img = zhizhen
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.medianBlur(gray, 5)

        #for testing, output gray image
        cv2.imwrite('%s-%s-bw.%s' %(name,gauge_number, file_type),gray)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
         
        a, b, c = circles.shape
        x,y,r = self.avg_circles(circles, b)

        #画圆和圆心
        cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  
        cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  

        #画读数，每个间隔，
        separation = 10.0 
        interval = int(360 / separation)
        p1 = np.zeros((interval,2))  
        p2 = np.zeros((interval,2))
        p_text = np.zeros((interval,2))
        for i in range(0,interval):
            for j in range(0,2):
                if (j%2==0):
                    p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
                else:
                    p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
        text_offset_x = 10
        text_offset_y = 5
        for i in range(0, interval):
            for j in range(0, 2):
                if (j % 2 == 0):
                    p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                    p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
                else:
                    p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                    p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

        #add the lines and labels to the image
        for i in range(0,interval):
            cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
            cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

        cv2.imwrite('%s-calibration-%s.%s' % (name,gauge_number, file_type), img)

        #需要配置这个表盘的最大 最小，以及单位
        min_angle = Config.Minangle
        max_angle = Config.Maxangle
        min_value = Config.Minvalue
        max_value = Config.Maxvalue
        units = Config.units

        return min_angle, max_angle, min_value, max_value, units, x, y, r

    def get_current_value(self,img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type,name):


        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #比较模糊的，调高对比度
        #50cm 原图
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        #50cm 模糊2像素
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))


        #50cm 模糊3像素
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))


        #25cm 原图
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        #50cm 模糊3像素
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))

        #限制对比度的自适应阈值
        dst = clahe.apply(gray2)
        #原图一定屏蔽掉，模糊的要添加，原图添加，识别不了， 模糊的 不添加 识别不了
        #gray2 =dst

        thresh =155
        #1
        thresh =165
        #53
        thresh =160
        #55
        thresh =160
        #100cm
        thresh =160
        #100cm
        thresh =155

        #50cm 原图
        thresh =150

        #50cm 模糊
        thresh =140

        #25cm 模糊
        thresh =166

        maxValue = 255

        # apply thresholding which helps for finding lines
        th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)

        # find lines
        minLineLength = 10
        maxLineGap = 0
        lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

        #测试所有的线
        img1 = img.copy()
        for i in range(0, len(lines)):
          for x1, y1, x2, y2 in lines[i]:
             cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.imwrite('%s-lines-test-%s.%s' %(name,gauge_number, file_type), img1)

        # 移除所有的线超出半径，压线圆，
        final_line_list = []
    
        #指针线  指针点  和圆心的距离阈值
        #50cm 原图
        diff1LowerBound = 0.15  
        #50cm 模糊
        diff1LowerBound = 0.1

        #25cm 模糊
        diff1LowerBound = 0.1

        diff1UpperBound = 0.25

        diff2LowerBound = 0.0 
        diff2UpperBound = 1.0
        #以上四个默认 0.15 0.25  0.5  1.0

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                #final_line_list.append([x1, y1, x2, y2])
                diff1 = self.dist_2_pts(x, y, x1, y1)  # 点与圆心的距离
                diff2 = self.dist_2_pts(x, y, x2, y2)  
                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp
                # check if line is within an acceptable range
                if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                    line_length = self.dist_2_pts(x1, y1, x2, y2)
                    
                    final_line_list.append([x1, y1, x2, y2])

        #testing only, show all lines after filtering
        img2 = img.copy()
        for i in range(0,len(final_line_list)):
            x1 = final_line_list[i][0]
            y1 = final_line_list[i][1]
            x2 = final_line_list[i][2]
            y2 = final_line_list[i][3]
            cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite('%s-linesfliter-2-%s.%s' % (name,gauge_number, file_type), img2)

        # 输出第一个线
        img3 = img.copy()
        x1 = final_line_list[0][0]
        y1 = final_line_list[0][1]
        x2 = final_line_list[0][2]
        y2 = final_line_list[0][3]
        cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

        cv2.imwrite('%s-lines-2-%s.%s' % (name,gauge_number, file_type), img3)

        #find the farthest point from the center to be what is used to determine the angle
        dist_pt_0 = self.dist_2_pts(x, y, x1, y1)
        dist_pt_1 = self.dist_2_pts(x, y, x2, y2)
        if (dist_pt_0 > dist_pt_1):
            x_angle = x1 - x
            y_angle = y - y1
        else:
            x_angle = x2 - x
            y_angle = y - y2
        
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))

        #these were determined by trial and error
        res = np.rad2deg(res)
        if x_angle > 0 and y_angle > 0:  #in quadrant I
            final_angle = 270 - res
        if x_angle < 0 and y_angle > 0:  #in quadrant II
            final_angle = 90 - res
        if x_angle < 0 and y_angle < 0:  #in quadrant III
            final_angle = 90 - res
        if x_angle > 0 and y_angle < 0:  #in quadrant IV
            final_angle = 270 - res


        old_min = float(min_angle)
        old_max = float(max_angle)

        new_min = float(min_value)
        new_max = float(max_value)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        cv2.putText(img, '%f' %(new_value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)

        cv2.imwrite('%s-dushu-%s.%s' % (name,gauge_number, file_type), img)

        return new_value

    def dushu(self,x,inpimg):
        gauge_number = 1
        #name = 'gv0004' #1, 100, np.array([]), 200, 100,
        #name = 'gv0198'  #1, 10, np.array([]), 200, 50,
        #name = 'gauge-1'
        file_type='png'
        name ='56'
        file_type='png'
        
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        zhizhen = inpimg[int(x[1])-15:int(x[3])+15,int(x[0])-15:int(x[2])+15]
        #zhizhen = inpimg[int(x[1]):int(x[3]),int(x[0]):int(x[2])]
        #zhizhen = cv2.resize(zhizhen,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        val = ''
        try:
            min_angle, max_angle, min_value, max_value, units, x, y, r = self.calibrate_gauge(zhizhen,gauge_number,name, file_type)

            img = zhizhen
            shu = self.get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type,name)
            #print ("Current reading: %s %s" %(shu, units))
            if shu > Config.Minvalue and shu < Config.Maxvalue:
                val = str(f'{shu:.3f}')+" "+units
                return val
            else: return ' '
        except Exception as ret:
            #print(ret)
            return val
        # finally:
        #     return 'finally'
        

        

   	
