import numpy as np
import cv2

from time import sleep
import MySQLdb
from MySQLdb import errorcode

# db=MySQLdb.connect(passwd="moonpie",db="thangs")
# c=db.cursor()
# max_price=5
# c.execute("""SELECT spam, eggs, sausage FROM breakfast
#           WHERE price < %s""", (max_price,))


# Obtain connection string information from the portal
config = {
  'host':'oursystem.mysql.database.azure.com',
  'user':'user',
  'password':'pass',
  'database':'projectdb'
}

try:
   conn = MySQLdb.connect(**config)
   print("Connection established")
except MySQLdb.Error as err:
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