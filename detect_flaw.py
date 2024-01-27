# -*- coding: utf-8 -*-
import cv2
import argparse
import numpy as np
import pymysql
import datetime
from  config import  Config
import time

class yolov5():
    def __init__(self, yolo_type,names_path,model_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):

        with open(names_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')   ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        yolo_type = model_path
        self.net = cv2.dnn.readNet(yolo_type )
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # inference output
        outs = 1 / (1 + np.exp(-outs))   ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind+length, 0:2] = (outs[row_ind:row_ind+length, 0:2] * 2. - 0.5 + np.tile(self.grid[i],(self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind+length, 2:4] = (outs[row_ind:row_ind+length, 2:4] * 2) ** 2 * np.repeat(self.anchor_grid[i],h*w, axis=0)
            row_ind += length
        return outs


def Detect(img,names_path,model_path):
    
    net_type='yolov5s'
    confThreshold=0.5
    nmsThreshold=0.5
    objThreshold=0.5
    
    yolonet = yolov5(net_type,names_path,model_path, confThreshold=confThreshold, nmsThreshold=nmsThreshold, objThreshold=objThreshold)
    srcimg = img
    dets = yolonet.detect(srcimg)
    srcimg = yolonet.postprocess(srcimg, dets)

    return srcimg,dets

def DecFlaw():
    try:
        db = pymysql.connect(host=Config.mysql_host, port=Config.mysql_port, user=Config.mysql_user, \
        password=Config.mysql_password, database=Config.mysql_database,charset='utf8mb4')
        cursor = db.cursor()

        cap=cv2.VideoCapture(0)

        i=0
        while (cap.isOpened() ):

            ret,raw_frame=cap.read()


            names_path = './coco.liehen.names'
            model_path = './yolov5s.liehen.onnx'
            dishui_frame,bbox = Detect(raw_frame,names_path,model_path)
            time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

            cv2.imwrite("frame_flaw.jpg",dishui_frame)
            filei = open("frame_flaw.jpg",'rb')
            dataimg = filei.read()

            sql_bj = "INSERT INTO diagnosis_history(task_name,fault_id,image,time_start) values(%s,%s,%s,%s);"
            cursor.execute(sql_bj,('裂痕检测',0,dataimg,time_now))
            db.commit()

            if len(bbox)!=0:
                time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

                sql_bj = "INSERT INTO diagnosis_list(task_name,fault_id,image,time_start) values(%s,%s,%s,%s);"
                cursor.execute(sql_bj,('裂痕检测',1,dataimg,time_now))
                db.commit()

            time.sleep(1)
            if i==0:break
            i+=1


    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Code Stopped by User")

    del dataimg
    cv2.destroyAllWindows()
    cap.release()
    db.close()



if __name__ == "__main__":
    try:
        db = pymysql.connect(host=Config.mysql_host, port=Config.mysql_port, user=Config.mysql_user, \
        password=Config.mysql_password, database=Config.mysql_database,charset='utf8mb4')
        cursor = db.cursor()

        cap=cv2.VideoCapture(0)

        i=0
        while (cap.isOpened() ):

            ret,raw_frame=cap.read()


            names_path = './coco.dishui.names'
            model_path = './yolov5s.dishui.onnx'
            dishui_frame,bbox = Detect(raw_frame,names_path,model_path)
            time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

            cv2.imwrite("dishui_frame.jpg",dishui_frame)
            filei = open("dishui_frame.jpg",'rb')
            dataimg = filei.read()

            sql_bj = "INSERT INTO diagnosis_history(task_name,fault_id,image,time_start) values(%s,%s,%s,%s);"
            cursor.execute(sql_bj,('漏水检测',0,dataimg,time_now))
            db.commit()

            if len(bbox)!=0:
                time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

                sql_bj = "INSERT INTO diagnosis_list(task_name,fault_id,image,time_start) values(%s,%s,%s,%s);"
                cursor.execute(sql_bj,('漏水检测',1,dataimg,time_now))
                db.commit()

            time.sleep(1)
            if i==0:break
            i+=1


    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Code Stopped by User")

    del dataimg
    cv2.destroyAllWindows()
    cap.release()
    db.close()