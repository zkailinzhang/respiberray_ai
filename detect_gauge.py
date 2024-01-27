
import argparse
import sys
import time
import pymysql
import datetime

from pathlib import Path
from config import Config
import cv2
import torch
import torch.backends.cudnn as cudnn

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
from zhenreader import Reader
from config import Config

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box,plot_one_box2,plot_one_box32,plot_one_box31,plot_one_box311
from utils.torch_utils import select_device, load_classifier, time_sync

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

readzhen = Reader()



@torch.no_grad()
def run(weights='./yibiao_yolov5.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam 
        ):

    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000 # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project='runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
       
    save_img = not nosave and not source.endswith('.txt') 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir).mkdir(parents=True, exist_ok=True)  
 
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    w = weights[0] if isinstance(weights, list) else weights

    stride, names = 64, []  

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  
    names = model.names 
       
    imgsz = check_img_size(imgsz, s=stride)  

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    bs = 1  
    vid_path, vid_writer = [None] * bs, [None] * bs

    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  #归一化
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]

        # NMS非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
         
        # img 是喂入模型的，img有做预处理，
        # im0 原始图片
        # imc 原始图像 的复制

        rst =""

        for i, det in enumerate(pred):  
            if webcam:  
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)  
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            imc = im0.copy() if save_crop else im0  
            if len(det):
                # 图像还原
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
 
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  

                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop : 
                        c = int(cls) 
                        
                        #读数
                        dushuimg = im0.copy()
                        
                        if  source.split('.')[-1].lower() in VID_FORMATS and conf >= Config.confi:
                            #plot_one_box(xyxy, im0,  color=colors(c, True), line_thickness=line_thickness)

                            rst = readzhen.dushu(xyxy, dushuimg)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {rst}')
                            
                            #plot_one_box2(xyxy, im0, label=label,dushu=rst, color=colors(c, True), line_thickness=line_thickness)

                        if  source.split('.')[-1].lower() in IMG_FORMATS:
                            #plot_one_box311(xyxy, im0,  color=colors(c, True), line_thickness=line_thickness)
                            # cv2.namedWindow("zhizhendushu", 0)
                            # cv2.imshow("zhizhendushu",im0)
                            # cv2.waitKey(2)
                            rst = readzhen.dushu(xyxy, dushuimg)
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {rst}')
                            
                            plot_one_box32(xyxy, im0, label=label,dushu=rst, color=colors(c, True), line_thickness=line_thickness)

                        
                        #可视化
                        #cv2.namedWindow("zhizhendushu", 0)
                        #cv2.imshow("zhizhendushu",im0)
                        #cv2.waitKey(1)
                        #imgsva = str(i)+'.jpg'
                        #cv2.imwrite(imgsva,im0)

            #要添加异常捕获，检测不到，正常保存， 读数不到，就不读，正常


    
    print(f"Results saved to {save_dir}{s}")
    return rst,im0



def DecGauge():
    try:
        db = pymysql.connect(host=Config.mysql_host, port=Config.mysql_port, user=Config.mysql_user, \
        password=Config.mysql_password, database=Config.mysql_database,charset='utf8mb4')
        cursor = db.cursor()

        cap=cv2.VideoCapture(0)

        i=0
        while (cap.isOpened() ):

            ret,raw_frame=cap.read()

            imgsva = "./yibiaos/" +str(i)+'.png'
            cv2.imwrite(imgsva,raw_frame)

            rst,yibiao_frame = run(weights='./yibiao_yolov5.pt', source=imgsva)
            

            time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')

            cv2.imwrite("frame_gauge.png",yibiao_frame)
            filei = open("frame_guage.pngs",'rb')
            dataimg = filei.read()



            if isinstance(rst,float):
                time_now = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')
                sql_bj = "INSERT INTO diagnosis_list(task_name,fault_id,data_value,image,time_start) values(%s,%s,%s,%s,%s);"
                cursor.execute(sql_bj,("表计检测",1,rst,dataimg,time_now))
                db.commit()
            else:
                sql_bj = "INSERT INTO diagnosis_history(task_name,fault_id,image,time_start) values(%s,%s,%s,%s,%s);"
                cursor.execute(sql_bj,("表计检测",0,dataimg,time_now))
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
