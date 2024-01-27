import numpy as np
import cv2
import mysql.connector
from mysql.connector import errorcode
from time import sleep
import serial



import MySQLdb
db=MySQLdb.connect(passwd="moonpie",db="thangs")
c=db.cursor()
max_price=5
c.execute("""SELECT spam, eggs, sausage FROM breakfast
          WHERE price < %s""", (max_price,))



          
# Obtain connection string information from the portal
config = {
  'host':'oursystem.mysql.database.azure.com',
  'user':'user',
  'password':'pass',
  'database':'projectdb'
}

try:
   conn = mysql.connector.connect(**config)
   print("Connection established")
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with the user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)

cursor = conn.cursor()
cursor.execute("CREATE TABLE if not exists Camera (img BLOB);")
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame = cap.read()[1]
cursor.execute("INSERT INTO Camera (img) VALUES %s",frame)
cap.release()



##CV2 to Base64
import cv2
 
def cv2_base64(image):
  base64_str = cv2.imencode('.jpg',image)[1].tostring()
  base64_str = base64.b64encode(base64_str)
  return base64_str 
 
 
##Base64 to CV2
import base64
import numpy as np
import cv2
 
def base64_cv2(base64_str):
  imgString = base64.b64decode(base64_str)
  nparr = np.fromstring(imgString,np.uint8) 
  image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
  return image





blob_value = open('image.jpg', 'rb').read()
sql = 'INSERT INTO Tab1(blob_field) VALUES(%s)'
args = (blob_value, )

cursor.execute (sql, args)
connection.commit()



sql = 'SELECT `blob_field` FROM `Tab1`'
cursor.execute(sql)
for row in cursor:
    blob_value = row[0]


















import mysql.connector

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insertBLOB(emp_id, name, photo, biodataFile):
    print("Inserting BLOB into python_employee table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='python_db',
                                             user='pynative',
                                             password='pynative@#29')

        cursor = connection.cursor()
        sql_insert_blob_query = """ INSERT INTO python_employee
                          (id, name, photo, biodata) VALUES (%s,%s,%s,%s)"""

        empPicture = convertToBinaryData(photo)
        file = convertToBinaryData(biodataFile)

        # Convert data into tuple format
        insert_blob_tuple = (emp_id, name, empPicture, file)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into python_employee table", result)

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

insertBLOB(1, "Eric", "D:\Python\Articles\my_SQL\images\eric_photo.png",
           "D:\Python\Articles\my_SQL\images\eric_bioData.txt")
insertBLOB(2, "Scott", "D:\Python\Articles\my_SQL\images\scott_photo.png",
           "D:\Python\Articles\my_SQL\images\scott_bioData.txt")