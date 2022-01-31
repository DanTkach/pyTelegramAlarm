import telebot
import cv2
import numpy as np

token = "5181674508:AAF21TJ1WVr7xCWXYW8MCgeGATqnz30OEq8"
chat_id = "430284875"

bot = telebot.TeleBot(token)
face=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)
winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
prev_frame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
current_frame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
next_frame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

def diffImg(f0, f1, f2):
    d1 = cv2.absdiff(f2, f1)
    d2 = cv2.absdiff(f1, f0)
    res=cv2.bitwise_and(d1, d2)
    d3 = np.ravel(res)
    d4=np.count_nonzero(d3)
    return d4,res

while True:
    frame=cam.read()[1]
    nzero,result=diffImg(prev_frame, current_frame, next_frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face.detectMultiScale(gray)
    print(faces)
    if nzero> 160000 and not(np.sum(faces)==0):
        _ret,frame=cam.read()
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('cam',frame)
        cv2.imwrite('1.png',frame)
        print("moving")
        bot.send_photo(chat_id, open('1.png', 'rb'))
        bot.send_message(chat_id,"Face detected")
        nzero=0
    cv2.imshow( winName, result)
    prev_frame = current_frame
    current_frame = next_frame
    next_frame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyWindow(winName)
        break