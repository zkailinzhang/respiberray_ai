#!/bin/bash
cd /home/pi/zkl/zhenghesmp
nohup /usr/bin/python3 -u /home/pi/zkl/zhenghesmp/hongwai.py  >>  /home/pi/zkl/zhenghesmp/log/serving.online.tempera.out 2>&1 &
