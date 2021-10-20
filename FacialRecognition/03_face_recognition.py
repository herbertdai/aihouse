''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 
import time
import datetime

from aip import AipSpeech
import pygame 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'daiwenyuan', 'zhangli', 'daijiayi', 'daijiayue', 'nainai'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

g_lastText = ''
g_lastId = ''
g_video_rec = 99 

def ttsbaidu(text):
    """ 你的 APPID AK SK """
    APP_ID = '24887102'
    API_KEY = 'ypUFaYjmLewLHIOwqop9kaXu'
    SECRET_KEY = 'tV38Hi5lELhel9xoGKLs9hrCxGWZSWbw'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    result  = client.synthesis('你好' + text, 'zh', 1, {
    'vol': 5,
})
    print("call baidu api")

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open(text+'.mp3', 'wb') as f:
            f.write(result)

import os

def playaudio(text):
    if (text == ''):
        return

    # not necessary to call baidu everytime
    if not os.path.exists(text + ".mp3"):
        ttsbaidu(text)

    print("play: " + text)
    pygame.mixer.init()
    pygame.mixer.music.load(text + ".mp3")
    pygame.mixer.music.set_volume(1)
    pygame.mixer.music.play()
    time.sleep(5)    

_SLEEP_ = 0


def isInTime(time1, time2):
    """ if now time is in a time area """
    # 范围时间
    start_time = datetime.datetime.strptime(str(datetime.datetime.now().date()) +  time1, '%Y-%m-%d%H:%M')
    # 开始时间
    print(start_time)
    end_time = datetime.datetime.strptime(str(datetime.datetime.now().date()) + time2, '%Y-%m-%d%H:%M')
    # 结束时间
    print(end_time)
    # 当前时间
    now_time = datetime.datetime.now()
    # 方法一：
    # 判断当前时间是否在范围时间内
    if start_time < now_time < end_time:
        print("是在这个时间区间内")
        return True
    else:
        return False


def notifySound(id):
    global _SLEEP_
    global g_video_rec
    txtToPlay = ''
    today = datetime.datetime.now().weekday() + 1
    
    if (g_video_rec > 20):
        g_video_rec = 0
        print("skip 20 video detections")
    else:
        g_video_rec += 1
        print(g_video_rec)
        return
    
    if (id == 'daiwenyuan'):
        playaudio('爸爸')

    elif (id == 'zhangli'):
        playaudio('妈妈，记得要多喝水 ')
    elif (id == 'daijiayi'):
        txtToPlay = '佳佳' + '今天是星期' + str(today)
        if (today == 4):
            if isInTime('6:00', '8:00'):
                txtToPlay += '，早上有英语课、语文课和数学课，还有《道德与法治》'
            elif isInTime('12:00', '14:00'):
                txtToPlay += '，今天下午有数学和体育'
    elif (id == 'unknown'):
        txtToPlay = ''

    playaudio(txtToPlay)
    time.sleep(_SLEEP_)


while True:

    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        notifySound(id)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    #cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
