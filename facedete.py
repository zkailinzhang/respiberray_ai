import cv2
#捕获相机
cap = cv2.VideoCapture("./2222.mp4")

while(True):
    #采集图片
    ret, frame = cap.read()

    # 创建haar人脸特征
    faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml") 
    #rgb转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#3
    # 检测人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5,5),
        flags = cv2.CASCADE_SCALE_IMAGE
    ) 
    print ("找到 {0} 张脸!".format(len(faces)))#5
    for (x, y, w, h) in faces:
        #画框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #6

    # 可视化
    cv2.imshow('人脸检测',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放相机
cap.release()
cv2.destroyAllWindows()