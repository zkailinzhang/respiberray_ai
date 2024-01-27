import numpy as np 
import base64
import cv2 
import MySQLdb
import pymysql
import PIL.Image
import time
import datetime

'''
尽量不用base64  

用二进制即可，blob 存mysql 读取
'''
path = 'd:\\123.png'
aa = open(path,'rb')

#写库成功
sql_bj = "INSERT INTO diagnosis_list (id,image) values (%s,%s)"
db = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='weersoon',charset='utf8mb4')
cursor = db.cursor()
#aa.read()  再read就没有了
cursor.execute(sql_bj,(1,aa.read(),))
db.commit()


#这样写就报错
sql_bj = "INSERT INTO %s (image) VALUES(%s);" %('diagnosis_list',aa.read())
cursor.execute(sql_bj)



#读取 再保存图像ok
sql = 'SELECT image FROM diagnosis_list where id=12;'
cursor.execute(sql)
blob_value =[]
for row in cursor:
    blob_value = row[0]
fo = open('xxxx.jpg','wb')
fo.write(blob_value)




img = cv2.imread('/home/zkl/Downloads/LCD.jpg')
ret, buf = cv2.imencode('.jpg',img)
jpgastext = base64.b64encode(buf)
print(jpgastext)
len(jpgastext)
#70348
np.shape(buf)
#(52761, 1)

aa = open(path,'rb')
aa = open('/home/zkl/Downloads/LCD.jpg','rb')
#<_io.BufferedReader name='/home/zkl/Downloads/LCD.jpg'>

aa.read()
#b'\xf..........'

#要加库，不然后面报错
db = MySQLdb.connect('localhost','root','123456')
cursor = db.cursor()

cursor
#<MySQLdb.cursors.Cursor at 0x7f6eac14e898>
#先创建库，create database vvideo;
db = MySQLdb.connect('localhost','root','123456','vvideo')




sql = 'SELECT * FROM `diagnosis_list`'

#1
cursor.execute(sql)

#设备报警历史表device_warning_history
sql_bj = "INSERT INTO %s (fault_id,time_start,time_end,data_value,fault_level,fault_type,\
image,device_id,task_id,task_name,device_name) \
VALUES(%s,'%s','%s',%s, %s,%s,%s,%s,%s,%s,'%s');" %(
                                        'diagnosis_list'
                                        ,1,timestamp,
                                        ,timestamp
                                        ,device_name
                                        ,'异常',
                                        warn_id,ID_fault,1)

start_row = (datetime.datetime.now()-datetime.timedelta(minutes=35)).strftime(f'%Y-%m-%d %H:%M:%S')
end_row = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')
sql_bj = "INSERT INTO %s (fault_id,time_start,time_end,image) VALUES(%s,'%s','%s',%s);" %('diagnosis_list',1,start_row,end_row,aa)
sql_bj = "INSERT INTO %s (fault_id,time_start,time_end) VALUES(%s,'%s','%s');" %('diagnosis_list',1,start_row,end_row)


mysql_insert_sentence = 'fault_id,time_start,time_end,image'

value = (system_id, i, cluster_result[i][0], cluster_result[i][1], cluster_result[i][2],
                                    cluster_result[i][3], cluster_result[i][4], cluster_result[i][5], cluster_result[i][6], 0)


start_row = (datetime.datetime.now()-datetime.timedelta(minutes=35)).strftime(f'%Y-%m-%d %H:%M:%S')
end_row = (datetime.datetime.now()).strftime(f'%Y-%m-%d %H:%M:%S')
value = (1,start_row,end_row,)

mysql.insert_data('diagnosis_list', mysql_insert_sentence, value)

#0  创建表
cursor.execute("CREATE TABLE if not exists Camera (img BLOB);")



cursor.execute("INSERT INTO Camera (img) VALUES(%s)",(aa,))

#要提交，不然mysql库里看不到
db.commit()


sql = 'SELECT `img` FROM `Camera`'

#1
cursor.execute(sql)


for row in cursor:
    blob_value = row[0]

fo = open('xxxx.jpg','wb')
fo.write(blob_value)


 sql 
        = 
        "SELECT name, ablob FROM justatest ORDER BY name"
       
 
            
        cursor.execute(sql)
       
 
            
        for 
        name, blob 
        in 
        cursor.fetchall( ):