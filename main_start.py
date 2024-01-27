# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
from apscheduler.schedulers.background import BackgroundScheduler
import socket
import pymysql
from config import Config
from detect_water import DecWater 
from hongwai import DetectWendu
# from detect_gauge import DecGauge
# from detect_oil import DecOil
# from detect_flaw import DecFlaw
# from detect_bolt import DecBolt

import os 

if __name__ == '__main__':
    # BackgroundScheduler: 适合于要求任何在程序后台运行的情况，当希望调度器在应用后台执行时使用
    scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
    # 采用非阻塞的方式

    # 采用cron的方式
    #scheduler.add_job(job, 'cron', day_of_week='fri', second='*/5')
    #scheduler.add_job(job, 'cron', second='*/5')
    #scheduler.add_job(job2, 'cron', second='*/5')

    # 漏油监测 0、
    # 漏水监测 1、
    # 螺栓监测 2、
    # 温度监测 3、
    # 表计监测 4、
    # 划痕监测 5


    # #获取计算机名称
    # hostname=socket.gethostname()
    # #获取本机IP
    # ip=socket.gethostbyname(hostname)
    # print(ip)

    ip=''
    try: 
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) 
        s.connect(('8.8.8.8',80)) 
        ip = s.getsockname()[0] 
    finally: 
        s.close() 
    print(ip) 

    db = pymysql.connect(host=Config.mysql_host, port=Config.mysql_port, user=Config.mysql_user, \
    password=Config.mysql_password, database=Config.mysql_database,charset='utf8mb4')
    cursor = db.cursor()
    #ip='192.168.1.156'
    sql_bj = "select task from device_task_relation where ip='%s'"%(ip) 

    cursor.execute(sql_bj)
    data = cursor.fetchall()
    db.commit()

    taskid = (data[0][0]).split(" ")

    if int(taskid[0]) ==True:
        print("ip: {},task:{}".format(ip,"漏油监测"))
        # scheduler.add_job(DecOil,trigger= 'cron', minute ='*/40')


    if int(taskid[1]) ==True:
        scheduler.add_job(DecWater,trigger= 'cron', minute ='*/10')
        print("ip: {},task:{}".format(ip,"漏水监测"))
    
    if int(taskid[2]) ==True:
        print("ip: {},task:{}".format(ip,"螺栓监测"))
        # scheduler.add_job(DecBolt,trigger= 'cron', hour ='*/5')
        pass

    if int(taskid[3]) ==True:
        print("ip: {},task:{}".format(ip,"温度监测"))
        scheduler.add_job(DetectWendu, trigger='cron', minute='*/3',args=[5])
        #pass

    if int(taskid[4]) ==True:
        print("ip: {},task:{}".format(ip,"表计监测"))
        #scheduler.add_job(DecGauge, trigger='cron', minute ="*/50")

    if int(taskid[5]) ==True:
        print("ip: {},task:{}".format(ip,"划痕监测"))
        # scheduler.add_job(DecFlaw, trigger='cron', hour ="*/20")
    
    scheduler.start()
    while True:
        time.sleep(1)
    # print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    # try:
    #     scheduler.start()
    # except (KeyboardInterrupt, SystemExit):
    #     scheduler.shutdown()