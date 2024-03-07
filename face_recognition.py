import cv2

#Code for Camera Enable 

# video_capture = cv2.VideoCapture(0)
# while True:
#     rt , video_data = video_capture.read()
#     cv2.imshow("Face Recognition Window",video_data)
#     if(cv2.waitKey(10) == ord("c")):
#         break

# video_capture.release()

#Code for Face Detection through camera

face_capture = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #for classifying Eyes,nose,mouth
video_capture = cv2.VideoCapture(0)
while True:
    rt , video_data = video_capture.read()
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    face = face_capture.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in face:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),4)
    cv2.imshow("Face Recognition Window",video_data)
    if(cv2.waitKey(10) == ord("c")):
        break

video_capture.release()
