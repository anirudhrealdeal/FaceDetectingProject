# Here we're gonna look at the very Basics of Face Detection
import cv2.cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('Videos/2.mp4')
cTime=0
pTime=0

mpFaceDetection = mp.solutions.face_detection
faceDetection= mpFaceDetection.FaceDetection(0.75) # To increase the accuracy of detection
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imgRGB= cv.cvtColor(image,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(image,detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            boundingBoxFromClass= detection.location_data.relative_bounding_box
            h,w,c= image.shape
            boundingBox = int(boundingBoxFromClass.xmin*w),int(boundingBoxFromClass.ymin*h),\
                          int(boundingBoxFromClass.width*w),int(boundingBoxFromClass.height * h)
            cv.rectangle(image,boundingBox, (0,255,0),2) # Here we are drawing ourselves
            cv.putText(image, f'{int(detection.score[0]*100)}%', (boundingBox[0],boundingBox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv.putText(image, f'FPS:{int(fps)}', (50, 90), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv.imshow("Image", image)
    cv.waitKey(10)



