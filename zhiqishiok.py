from typing import ValuesView
import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
from config import Config
import math
import pandas as pd


def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)



#P为线外一点，AB为线段两个端点
def getDist_P2L(PointP,Pointa,Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    #求直线方程
    A=0
    B=0
    C=0
    A=Pointa[1]-Pointb[1]
    B=Pointb[0]-Pointa[0]
    C=Pointa[0]*Pointb[1]-Pointa[1]*Pointb[0]
    #代入点到直线距离公式
    distance=0
    distance=(A*PointP[0]+B*PointP[1]+C)/math.sqrt(A*A+B*B)
    
    return distance

file_type='png'
name ='1'
gauge_number = 1

img = cv2.imread('./yuan/22.jpg')
img = cv2.imread('./TEST/1.png')
height, width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
#gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gray = cv2.medianBlur(gray, 5)

#for testing, output gray image
cv2.imwrite('%s-%s-bw.%s' %(name,gauge_number, file_type),gray)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, np.array([]), 100, 50, int(height*0.30), int(height*0.40))
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, np.array([]), 100, 50, int(height), int(height))


# average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
a, b, c = circles.shape
xc,yc,r = avg_circles(circles, b)

imgw = img.copy()

#draw center and circle
cv2.circle(imgw, (xc, yc), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
cv2.circle(imgw, (xc, yc), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

print("xc,yc,r,",xc,yc,r)


canny= cv2.Canny(gray, 100, 10)
cv2.imwrite('%s-canny-%s.%s' % (name, gauge_number,file_type), canny)

cv2.namedWindow("polar",cv2.WINDOW_NORMAL)
cv2.namedWindow("polar5",cv2.WINDOW_NORMAL)




ro,col,_=img.shape
cent=(int(col/2),int(ro/2))
max_radius = int(np.sqrt(ro**2+col**2)/2)
max_radius = 0.7*min(xc, yc)


polar=cv2.linearPolar(canny,(xc,yc),r,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
cv2.imshow('polar', polar)
cv2.imwrite('%s-polar-%s.%s' % (name, gauge_number,file_type), polar)

thresh =120
maxValue = 255
th, dst5 = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY_INV)
cv2.imshow('dst5', dst5)
canny5= cv2.Canny(dst5, 100, 10)
cv2.imshow('canny5', canny5)

polar5=cv2.linearPolar(dst5,(xc,yc),r,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
cv2.imshow('polar5', polar5)
cv2.imwrite('%s-polar5-%s.%s' % (name, gauge_number,file_type), polar5)
cv2.waitKey()



imgtt= img.copy()
'''
#画读数，每个间隔，
separation = 10.0 
interval = int(360 / separation)
p1 = np.zeros((interval,2))  
p2 = np.zeros((interval,2))
p_text = np.zeros((interval,2))
for i in range(0,interval):
    for j in range(0,2):
        if (j%2==0):
            p1[i][j] = xc + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
        else:
            p1[i][j] = yc + 0.9 * r * np.sin(separation * i * 3.14 / 180)
text_offset_x = 10
text_offset_y = 5
for i in range(0, interval):
    for j in range(0, 2):
        if (j % 2 == 0):
            p2[i][j] = xc + r * np.cos(separation * i * 3.14 / 180)
            p_text[i][j] = xc - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
        else:
            p2[i][j] = yc + r * np.sin(separation * i * 3.14 / 180)
            p_text[i][j] = yc + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

#add the lines and labels to the image
for i in range(0,interval):
    cv2.line(imgtt, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
    cv2.putText(imgtt, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

cv2.imwrite('%s-calibration-%s.%s' % (name,gauge_number, file_type), imgtt)
'''



separation= 10 #in degrees
interval = int(360/separation)
p3 = np.zeros((interval,2))  #set empty arrays
p4 = np.zeros((interval,2))

for i in range(0,interval):
    for j in range(0,2):
        if (j%2==0):
            #33  0.99  11  0.9  22 
            p3[i][j] = xc + 0.9 * r * np.cos(separation * i * np.pi / 180) #point for lines
        else:
            p3[i][j] = yc + 0.9 * r * np.sin(separation * i * np.pi / 180)


def region_of_interest(img, vertices):
    mask= np.zeros_like(img)

    match_mask_color= 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    #可以更改为 环
    cv2.imwrite('%s-mask-%s.%s' % (name, gauge_number,file_type), mask)

    masked_image= cv2.bitwise_and(img, mask)
    cv2.imwrite('%s-mask2-%s.%s' % (name, gauge_number,file_type), masked_image)
    return masked_image


canny= cv2.Canny(gray, 200, 20)
cv2.imwrite('%s-canny-%s.%s' % (name, gauge_number,file_type), canny)
region_of_interest_vertices= p3
cropped_image= region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))

cv2.imwrite('%s-crop-%s.%s' % (name, gauge_number,file_type), cropped_image)


cv2.namedWindow("contoursjpg",cv2.WINDOW_NORMAL)
contours3= img.copy()

maskpl= np.zeros_like(cropped_image)

contours, heirarchy= cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
int_cnt= []
for cnt in contours:
    area = cv2.contourArea(cnt)
    [x, y, w, h] = cv2.boundingRect(cnt)
    cpd = dist_2_pts(x+w/2,y+h/2,xc, yc)
    #33  4/4  11  3.5/4  22  
    if area<500 and int(cpd) <r*3.5/4 and int(cpd) > r*3/4:
        cv2.drawContours(contours3, cnt, -1, (255,0,0), 3)
        cv2.drawContours(maskpl, cnt, -1, 255, 3)
        int_cnt.append(cnt) 
        cv2.imshow('contoursjpg', contours3)
        cv2.waitKey(5)


cv2.imwrite('%s-contours3-%s.%s' % (name, gauge_number,file_type), contours3)



cv2.namedWindow("polar2",cv2.WINDOW_NORMAL)
ro,col,_=img.shape
cent=(int(col/2),int(ro/2))
max_radius = int(np.sqrt(ro**2+col**2)/2)
max_radius = 0.7*min(xc, yc)
polar=cv2.linearPolar(maskpl,(xc,yc),r,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
cv2.imshow('polar2', polar)
cv2.waitKey()



frth_quad_index=[]
thrd_quad_index=[]
#10 350
reference_zero_angle= 20
reference_end_angle= 340
min_angle=90
max_angle=270


frth_quad_angle=[]
thrd_quad_angle=[]


for i in range(len(int_cnt)):
    a= int_cnt[i]
    a= a.reshape(len(a),2)
    a= pd.DataFrame(a)        
    x1= a.iloc[:,0].mean()
    y1= a.iloc[:,1].mean()
    
    xlen= x1-xc
    ylen= yc-y1
    
    #Taking arc-tan of ylen/xlen to find the angle
    res= np.arctan(np.divide(float(ylen), float(xlen)))
    res= np.rad2deg(res)
    print('xlen,xlen: ',xlen,ylen,res)
    
    if xlen<0 and ylen<0:
        res= np.arctan(np.divide(float(abs(ylen)), float(abs(xlen))))
        res= np.rad2deg(res)
        final_start_angle= 90-res
        #print(i , final_angle)
        frth_quad_index.append(i)
        frth_quad_angle.append(final_start_angle)
        #
        if final_start_angle> reference_zero_angle:
            if final_start_angle<min_angle:
                min_angle= final_start_angle
        
    elif xlen>0 and ylen<0:
        res= np.arctan(np.divide(float(abs(ylen)), float(abs(xlen))))
        res= np.rad2deg(res)
        final_end_angle= 270+res
        thrd_quad_index.append(i)
        thrd_quad_angle.append(final_end_angle)
        #print(i , res)
        if final_end_angle<reference_end_angle:
            if final_end_angle>max_angle:
                max_angle= final_end_angle
               
print(f'Zero reading corresponds to {min_angle}')
print(f'End reading corresponds to {max_angle}')

#print('final_start_angle,frth_quad_index',final_start_angle,frth_quad_index,
#frth_quad_angle,final_end_angle,thrd_quad_index,thrd_quad_angle)

#Zero reading corresponds to 29.05494071131895
#End reading corresponds to 349.08894115677197
#只是简单的归类，没有严格排序，所以归到第一和第四象限后，再把角度排序，分别排序，从小到大，然后计算两两差值，，
# 偏差大的，索引，，原始数组，索引加1，，取出角度
#Zero reading corresponds to 29.05494071131895
#End reading corresponds to 349.08894115677197


frth_angle_ = frth_quad_angle.copy()
frth_angle_.sort(reverse=False)
thrd_angle_ = thrd_quad_angle.copy()
thrd_angle_.sort(reverse=True)

frth_sub = [frth_angle_[i+1]-frth_angle_[i] for i in range(len(frth_angle_)-1) ]     
thrd_sub = [thrd_angle_[i+1]-thrd_angle_[i] for i in range(len(thrd_angle_)-1) ] 


min_angle_ = frth_angle_[np.argmax (np.array(frth_sub))+1]

max_angle = thrd_angle_[np.argmin (np.array(thrd_sub))+1]
print(f'Zero reading corresponds to {min_angle}')
print(f'End reading corresponds to {max_angle}')

thresh =165

maxValue = 255

# mgrey_img = cv2.medianBlur(gray2, 5)

# apply thresholding which helps for finding lines
th, dst2 = cv2.threshold(cropped_image, thresh, maxValue, cv2.THRESH_BINARY_INV)

# for testing, show image after thresholding
cv2.imwrite('%s-tempdst2-%s.%s' % (name, gauge_number,file_type), dst2)

# find lines
minLineLength = 100
maxLineGap = 0
lines = cv2.HoughLinesP(image=cropped_image, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=10)  # rho is set to 3 to detect more lines, easier to get more then filter them out later



#for testing purposes, show all found lines
img11 = img.copy()
for i in range(0, len(lines)):
  for x1, y1, x2, y2 in lines[i]:
     cv2.line(img11, (x1, y1), (x2, y2), (0, 255, 0), 2)
     cv2.imwrite('%s-lines-test-%s.%s' %(name,gauge_number, file_type), img11)

# remove all lines outside a given radius
final_line_list = []
#print "radius: %s" %r

diff1LowerBound = 0.05 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
diff1UpperBound = 0.25
diff2LowerBound = 0.05#diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
diff2UpperBound = 0.5
for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        #final_line_list.append([x1, y1, x2, y2])
        diff1 = dist_2_pts(xc, yc, x1, y1)  # x, y is center of circle
        diff2 = dist_2_pts(xc, yc, x2, y2)  # x, y is center of circle
        #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
        if (diff1 > diff2):
            temp = diff1
            diff1 = diff2
            diff2 = temp
        # check if line is within an acceptable range
        if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
            line_length = dist_2_pts(x1, y1, x2, y2)
            
            distance = getDist_P2L((xc,yc),(x1, y1),(x2, y2))
            print("line_length: ",line_length,"r: ",r,"distance: ",distance)
            if line_length>0.4*r and distance>-20 and distance <10:
            # add to final list
                final_line_list.append([x1, y1, x2, y2,distance])

#testing only, show all lines after filtering
img22 = img.copy()
for i in range(0,len(final_line_list)):
    x1 = final_line_list[i][0]
    y1 = final_line_list[i][1]
    x2 = final_line_list[i][2]
    y2 = final_line_list[i][3]
    cv2.line(img22, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('%s-linesfliter-2-%s.%s' % (name,gauge_number, file_type), img22)


#选择线 比较长的
# assumes the first line is the best one
img33 = img.copy()
final_line_list = final_line_list[np.argmax(np.array(final_line_list)[:,-1])]

x1 = final_line_list[0]
y1 = final_line_list[1]
x2 = final_line_list[2]
y2 = final_line_list[3]
cv2.line(img33, (x1, y1), (x2, y2), (0, 255, 0), 2)

#for testing purposes, show the line overlayed on the original image
#cv2.imwrite('gauge-1-test.jpg', img)
cv2.imwrite('%s-lines-2-%s.%s' % (name,gauge_number, file_type), img33)

#find the farthest point from the center to be what is used to determine the angle
dist_pt_0 = dist_2_pts(xc, yc, x1, y1)
dist_pt_1 = dist_2_pts(xc, yc, x2, y2)

y_angle=0
x_angle=0
if (dist_pt_0 > dist_pt_1):
    x_angle = x1 - xc
    y_angle = yc - y1
else:
    x_angle = x2 - xc
    y_angle = yc - y2


x_angle = (x1+x2)/2.0 - xc
y_angle = yc -  (y1+y2)/2.0


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

min_value=0.0
max_value=1.6
new_min = float(min_value)
new_max = float(max_value)

old_value = final_angle

old_range = (old_max - old_min)
new_range = (new_max - new_min)
new_value = (((old_value - old_min) * new_range) / old_range) + new_min

cv2.putText(img, '%f' %(new_value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)

cv2.imwrite('%s-dushu-%s.%s' % (name,gauge_number, file_type), img)


print('dushu',new_value)