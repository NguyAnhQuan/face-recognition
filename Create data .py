import cv2 
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

id=1
if id == 1:
    print(0)
    for i in range(1,6):
        for j in range (1,21):
            filename = r"D:\test face detect\Code main\Code\data\anh." + str(i) + "." + str(j) + ".jpg"
            frame = cv2.imread(filename)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                if not os.path.exists(r'D:\test face detect\Code main\Code\data\dataset'):
                    os.makedirs(r'D:\test face detect\Code main\Code\data\dataset')
                cv2.imwrite(r'D:\test face detect\Code main\Code\data\dataset\anh'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])
if id == 2:
    cap = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    sampleNum = 0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fa = detector.detectMultiScale(gray, 1.1, 5)
        for(x,y,w,h) in fa:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
            if not os.path.exists('dataset2'):
                os.makedirs('dataset2')
            sampleNum+=1
            cv2.imwrite('dataset2/anh'+str(1)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
        cv2.imshow('frame',frame)
        if sampleNum > 19:
            break;
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()



# chụp ảnh và hiện số ảnh chụp sau mỗi 0.5s (nguy cơ bị nhận diện nhầm(mũi, mắt hoặc đồ vâtj xong quanh))
# if id == 2:
#     cap = cv2.VideoCapture(0)
#     detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     sampleNum = 0
#     last_capture_time = 0

#     while True:
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.1, 5)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             if not os.path.exists('dataset2'):
#                 os.makedirs('dataset2')
#             current_time = cv2.getTickCount()
#             if (current_time - last_capture_time) / cv2.getTickFrequency() >= 0.5:
#                 sampleNum += 1
#                 cv2.imwrite('dataset2/anh' + str(1) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
#                 last_capture_time = current_time
#                 cv2.putText(frame, f'Samples taken: {sampleNum}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow('frame', frame)
#         if sampleNum > 19:
#             break

#         cv2.waitKey(1)  # Wait for a short time and update the display

#     cap.release()
#     cv2.destroyAllWindows()
