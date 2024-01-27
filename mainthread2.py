from threading import Thread
from collections import deque
from multiprocessing import Process
import cv2



def producer(cap, q):
    while True:
        # print('producer execuation')
        if cap.isOpened():
            ret, img = cap.read()
            q.append(img)


def consumer(camera_index, outVideo, q):
    print("Start to capture and save video of camera {}...".format(camera_index))
    while True:
        if len(q) == 0:
            pass
        else:
            img = q.pop()
            # print('consumer execuation')
            img_res = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
            cv2.namedWindow("camera {}".format(camera_index),0)
            outVideo.write(img)
            cv2.imshow("camera {}".format(camera_index), img_res)
            cv2.waitKey(1)


def multithread_run(camera_index, url):
    # get size and fps of video
    width = 2560
    height = 1920
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

    # create VideoWriter for saving
    outVideo = cv2.VideoWriter('Video_save_{}.avi'.format(camera_index), fourcc, fps, (width, height))
    q = deque(maxlen=1)
    cap = cv2.VideoCapture(url)
    p1 = Thread(target=producer, args=(cap, q))
    c1 = Thread(target=consumer, args=(camera_index, outVideo, q))
    p1.start()
    c1.start()
    p1.join()
    c1.join()



if __name__ == "__main__":
    processes = []
    nloops = range(2)
	
	#Assume that the two camera IPs are 192.168.9.151 and 192.168.9.152
    url = 'rtsp://10.180.9.{}:554'
    # url = 'rtsp://admin:12345@10.180.9.122'
    camera_index = 151

    for i in nloops:
        t = Process(target=multithread_run, args=(camera_index + i, url.format(camera_index + i)))
        processes.append(t)

    for i in nloops:
        processes[i].start()

    for i in nloops:
        processes[i].join()




import cv2
import numpy as np

videoLeft = cv2.VideoCapture(0)
videoRight = cv2.VideoCapture(1)

width = (int(videoLeft.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(videoLeft.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while (videoLeft.isOpened() and videoRight.isOpened()):
    retLeft, frameLeft = videoLeft.read()
    retRight, frameRight = videoRight.read()
    if(retLeft and retRight):
        frameLeft = cv2.resize(frameLeft, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameRight = cv2.resize(frameRight, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        frameUp = np.hstack((frameLeft, frameRight))
        #再模拟2个窗口，组成4个窗口
        frameUp = np.hstack((frameLeft, frameRight))
        frame = np.vstack((frameUp, frameUp))

        cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

videoLeft.release()
videoRight.release()