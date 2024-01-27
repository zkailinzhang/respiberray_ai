#!/bin/bash
cd /home/pi/zkl/zhenghesmp
nohup /usr/bin/python3 -u /home/pi/zkl/zhenghesmp/detect_water.py  >>  /home/pi/zkl/zhenghesmp/log/serving.online.water.out 2>&1 &
