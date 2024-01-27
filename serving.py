# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np 
from flask import Flask, jsonify, request
import pickle
import statsmodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import pyplot as plt
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
import wget
import json
import requests
import subprocess
from enum import Enum
import redis 
import happybase
from concurrent.futures import ThreadPoolExecutor
import traceback
from config import Config,MyLogging
import math
import time
import itertools
from sklearn.mixture import GaussianMixture as GMM 
#from sklearn.externals import joblib
import joblib
from wsgiref.simple_server import make_server

import numpy as np
import warnings
import datetime


executor = ThreadPoolExecutor(16)

pathcwd = os.path.dirname(__file__)
app = Flask(__name__)



header = {'Content-Type': 'application/json','Accept': 'application/json'} 

local_path_data =''
cancel_diagnosis = {}
cancel_predict = {}


myLogger = MyLogging()

@app.route('/train_db', methods=['POST'])
def train_db():
    try:
        
        request_json = request.get_json()
        
        model_id = request_json["model_id"]
        device_name = request_json["device_name"]
        system_id = request_json["device_id"]
        versionid = request_json["model_version"]
        kks = request_json["kks"]
        interval = request_json["interval"]
        start_row = request_json["start_row"]
        end_row =  request_json["end_row"]
        
        train_future = executor.submit(train_db_task,model_id,versionid,system_id,kks,interval,start_row,end_row)
        
        myLogger.write_logger("******trainingdb 模型DB开始训练modelid {}".format(model_id))
        
        resp = jsonify({
                'status': True,
                'message': '-->模型DB开始训练',
                "address":"train_db"
        })
        resp.status_code = 200
        return resp


    except Exception as e:
        myLogger.write_logger("******trainingdb modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "python训练DB预处理异常",
        "modelId": model_id,
        "modelVersion":versionid,
        "address":"train_db"

        }
        requests.post(Config.java_host_train_db, \
                    data = json.dumps(message),\
                    headers= header)



def train_db_task(model_id,versionid,system_id,kks,interval,start_row,end_row):
    
    try:
        
        cluster = ClusterModule(Config,system_id,kks,interval,start_row,end_row,'limit_list')
        cluster.run()
        
        message = {
            'status': True,
            'message': "python训练DB完成",
            "modelId": model_id,
            "modelVersion":versionid,
            "address":"train_db"
        }
        myLogger.write_logger("******train_task db finished modelid {} ".format(model_id))

        requests.post(Config.java_host_train_db, \
                        data = json.dumps(message),\
                        headers= header)

    except Exception as e:
        myLogger.write_logger("******training db modelid {},excep:{}".format(model_id,e))
        message = {
        'status': False,
        'message': "python训练DB异常",
        "modelId": model_id,
        "modelVersion":versionid,
        "address":"train_db"

        }
        requests.post(Config.java_host_train_db, \
                    data = json.dumps(message),\
                    headers= header)




if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8484)#, debug=True)
    #server = make_server("127.0.0.1",app)
    #server.serve_forever()