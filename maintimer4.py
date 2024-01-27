import time
from apscheduler.schedulers.background import BackgroundScheduler


def job():
    print("job1",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

def job2():
    print("\njob2",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

if __name__ == '__main__':
    # BackgroundScheduler: 适合于要求任何在程序后台运行的情况，当希望调度器在应用后台执行时使用
    scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
    # 采用非阻塞的方式

    # 采用cron的方式
    #scheduler.add_job(job, 'cron', day_of_week='fri', second='*/5')
    scheduler.add_job(job, 'cron', second='*/5')
    scheduler.add_job(job2, 'interval', seconds=1)
    '''
    year (int|str) – 4-digit year
    month (int|str) – month (1-12)
    day (int|str) – day of the (1-31)
    week (int|str) – ISO week (1-53)
    day_of_week (int|str) – number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)
    hour (int|str) – hour (0-23)
    minute (int|str) – minute (0-59)
    econd (int|str) – second (0-59)
            
    start_date (datetime|str) – earliest possible date/time to trigger on (inclusive)
    end_date (datetime|str) – latest possible date/time to trigger on (inclusive)
    timezone (datetime.tzinfo|str) – time zone to use for the date/time calculations (defaults to scheduler timezone)
        
    *    any    Fire on every value
    */a    any    Fire every a values, starting from the minimum
    a-b    any    Fire on any value within the a-b range (a must be smaller than b)
    a-b/c    any    Fire every c values within the a-b range
    xth y    day    Fire on the x -th occurrence of weekday y within the month
    last x    day    Fire on the last occurrence of weekday x within the month
    last    day    Fire on the last day within the month
    x,y,z    any    Fire on any matching expression; can combine any number of any of the above expressions
    '''
    scheduler.start()
    
    # 其他任务是独立的线程
    # while True:
    #     print('main-start:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    #     time.sleep(2)
    #     print('main-end:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        
        
        
        
        
'''
main-start: 2021-12-15 09:19:40
main-end: 2021-12-15 09:19:42
main-start: 2021-12-15 09:19:42
main-end: 2021-12-15 09:19:44
main-start: 2021-12-15 09:19:44
job1 2021-12-15 09:19:45

job2 2021-12-15 09:19:45
main-end: 2021-12-15 09:19:46
main-start: 2021-12-15 09:19:46
main-end: 2021-12-15 09:19:48
main-start: 2021-12-15 09:19:48
job1 2021-12-15 09:19:50

job2 2021-12-15 09:19:50
main-end: 2021-12-15 09:19:50
main-start: 2021-12-15 09:19:50
main-end: 2021-12-15 09:19:52
main-start: 2021-12-15 09:19:52
main-end: 2021-12-15 09:19:54
main-start: 2021-12-15 09:19:54

job1 2021-12-15 09:19:55

job2 2021-12-15 09:19:55
main-end: 2021-12-15 09:19:56
main-start: 2021-12-15 09:19:56

'''