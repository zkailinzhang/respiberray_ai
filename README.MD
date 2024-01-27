
表字段  字典  再细化，
写表 细化
深度模型检测的，坐标要不要传，

周期调度， crontab

树莓派 命令行输入crontab -e
直接编辑，输入
*/8 * * * * /bin/bash /home/pi/zkl/zhenghesmp/start_dishui.sh
*/15 * * * * /bin/bash /home/pi/zkl/zhenghesmp/start_wendu.sh


以上 表示每8分钟

然后退出，命令行执行 周期调度开启，
service cron start

service cron stop


device_task_relation


漏油监测 0、
漏水监测 1、
螺栓监测 2、
温度监测 3、
表计监测 4、
划痕监测 5

传入“0 0 0 1 0” 
可以同时开启多个