
import cv2
import numpy as np 

def rgb2hsv(r, g, b): 
    
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn =  min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    s = round(s * 100)
    v = round(v * 100)
    return (h, s, v)


def extract_red_area(image):
    width = image.shape[0]
    height = image.shape[1]
    
    B = G = R =  H =  S =  V = 0.0
    redmat = np.zeros((width,height), dtype=np.int)
    for x in range(height):
        for y in range(width):
            B,G,R = image[y,x,0],image[y,x,1],image[y,x,2]
            (h, s, v) =  rgb2hsv(B, G,R )
            if((x==78 and y==80) or(x==94 and y==80) or(x==86 and y==78)or(x==159 and y==89)or(x==73 and y==77)or(x==176 and y==90)or(x==82 and y==80)or(x==141 and y==86)\
                or(x==143 and y==84)or(x==127 and y==84)or(x==156 and y==88)or(x==162 and y==89)or(x==177 and y==90)\
                    or(x==141 and y==86)or(x==146 and y==88)or(x==142 and y==85)or(x==145 and y==86)or(x==143 and y==83)):
                print(h, s, v)
            if (((h>=312 and h<= 360) and (s>=30 and s<=50) and (v>60 and v<80)) or ((h>=8 and h<= 30) and (s>=30 and s<=60) and (v>60 and v<80))):
                redmat[y,x]=255
    return redmat

def extract_red_area2(image):
    width = image.shape[0]
    height = image.shape[1]
    
    B = G = R =  H =  S =  V = 0.0
    redmat = np.zeros((width,height), dtype=np.int)
    for x in range(height):
        for y in range(width):
            h, s, v = image[y,x]
            if ((h>=90 and h<= 180) and (s>=10 and s<=130) and (v>165 and v<180)):
                redmat[y,x]=255
    return redmat


def luosuandet(image1,)
              

    image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    redmat = extract_red_area(image)
    redmat = redmat.astype(np.uint8)

    ret, thresh = cv2.threshold(redmat,  130, 255,  cv2.THRESH_BINARY) 

    kernel = np.ones((1, 3), np.uint8)
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, anchor=(1, 0), iterations=5)
    contours, hierarchy1 = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
 
    rstangle=[]
    for i,cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)#这里得到的是旋转矩形
        box = cv2.boxPoints(rect)#得到端点
        box = np.int0(box)#向下取整

        cv2.circle(image1,(int(rect[0][0]),int(rect[0][1])),2,(0,255,0),1)
        for j in range(4):
            cv2.line(image1,tuple(box[j]),tuple(box[(j+1)%4]),(255,0,0),1,8)
        rstangle.append(rect[2])
        print("角度：",rect[2])

    return rstangle




#判断角度差，



if __name__ == "__main__":
    image1 = cv2.imread("luosuan.jpg", cv2.IMREAD_UNCHANGED)   

    image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    cv2.imshow("0",image)



    redmat = extract_red_area(image)
    redmat = redmat.astype(np.uint8)
    cv2.imshow("1",redmat)

    
    # 2、二值化 ret:暂时就认为是设定的thresh阈值，thresh：二值化的图像
    ret, thresh = cv2.threshold(redmat,  # 转换为灰度图像,
        130, 255,  # >130的=255  否则=0
        cv2.THRESH_BINARY)  # 黑白二值化
    cv2.imshow("2",thresh)

    # 3、搜索轮廓: 返回：图像，轮廓，轮廓的层析结构        输入:图像（二值图像），轮廓检索方式，轮廓近似方法
    # contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # # cv2.RETR_EXTERNAL---只检测外轮廓     # cv2.RETR_LIST---检测的轮廓不建立等级关系
    # # cv2.RETR_CCOMP-建立两个等级轮廓，上一层为外边界，里层为内孔边界信息  # cv2.RETR_TREE----建立一个等级树结构的轮廓

    # # cv2.CHAIN_APPROX_NONE---存储所有边界点      # cv2.CHAIN_APPROX_SIMPLE---压缩垂直、水平、对角方向，只保留端点
    # # cv2.CHAIN_APPROX_TX89_L1---使用teh - Chini近似算法     # cv2.CHAIN_APPROX_TC89_KCOS---使用teh - Chini近似算法
    # cv2.drawContours(image1,contours,-1,(0,0,255),1)  
    # cv2.imshow("img1", image1)  





    kernel = np.ones((1, 3), np.uint8)
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, anchor=(1, 0), iterations=5)
    contours, hierarchy1 = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    # cv2.drawContours(image1,contours,-1,(0,0,255),1)  
    # cv2.imshow("img", image1)  


    rstangle=[]
    for i,cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)#这里得到的是旋转矩形
        box = cv2.boxPoints(rect)#得到端点
        box = np.int0(box)#向下取整

        cv2.circle(image1,(int(rect[0][0]),int(rect[0][1])),2,(0,255,0),1)
        for j in range(4):
            cv2.line(image1,tuple(box[j]),tuple(box[(j+1)%4]),(255,0,0),1,8)
        rstangle.append(rect[2])
        print("角度：",rect[2])

    if abs(rstangle[0] -rstangle[1]) >5.:
        print("螺栓检测：松动")
    else:
        print("螺栓检测：正常")

    cv2.imshow("img2", image1)
    cv2.imwrite("luoshuan.rst.jpg",image1)  
    cv2.waitKey()



'''

if len(contours)!=0:
    
    num = 0
    min_size =20
    max_size = 500
    hierarchy = hierarchy1[0] # get the actual inner list of hierarchy descriptions
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        if currentHierarchy[3]>0:
            del contours[num]
            del hierarchy[num]
        else:
            if(cv2.contourArea(currentContour) > max_size and cv2.contourArea(currentContour) < min_size) :
                del contours[num]
                del hierarchy[num]
        num+=1

    for i,cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)#这里得到的是旋转矩形
        box = cv2.BoxPoints(rect)#得到端点
        box = np.int0(box)#向下取整
        cv2.circle(image,(rect[0].x,rect[0].y),2,(0,255,255),-1)
        for j in range(4):
            cv2.line(image,box[j],box[(j+1)%4],(255,0,255),1,8)
            
        
        
    cv2.imshow("ss",image)
    cv2.waitKey()
  

# 获取最小外接矩阵，中心点坐标，宽高，旋转角度
rect = cv2.minAreaRect(points)
# 获取矩形四个顶点，浮点型
box = cv2.boxPoints(rect)
# 取整
box = np.int0(box)
# 获取四个顶点坐标
left_point_x = np.min(box[:, 0])
right_point_x = np.max(box[:, 0])
top_point_y = np.min(box[:, 1])
bottom_point_y = np.max(box[:, 1])
 
left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
# 上下左右四个点坐标
vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]])
————————————————
版权声明：本文为CSDN博主「Mein_Augenstern」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Maisie_Nan/article/details/105833892




# For each contour, find the bounding rectangle and draw it
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x,y,w,h = cv2.boundingRect(currentContour)
    if currentHierarchy[2] < 0:
        # these are the innermost child components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    elif currentHierarchy[3] < 0:
        # these are the outermost parent components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


    # create an empty mask
mask = np.zeros(cell1.shape[:2],dtype=np.uint8)

    # loop through the contours
for i,cnt in enumerate(contours):
            # if the contour has no other contours inside of it
    if hierarchy[0][i][2] == -1 :
                    # if the size of the contour is greater than a threshold
       if  cv2.contourArea(cnt) > 10000:
             cv2.drawContours(mask,[cnt], 0, (255), -1)  
'''