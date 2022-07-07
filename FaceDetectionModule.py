import cv2.cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # To increase the accuracy of detection
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self,image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        boundingBoxes=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBoxFromClass = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                boundingBox = int(boundingBoxFromClass.xmin * w), int(boundingBoxFromClass.ymin * h), \
                              int(boundingBoxFromClass.width * w), int(boundingBoxFromClass.height * h)
                boundingBoxes.append([id,boundingBox,detection.score])
                if draw:
                    image=self.fancyDraw(image,boundingBox)
                    cv.putText(image, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 20),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return image, boundingBoxes
    def fancyDraw(self,image, boundingBox, length=30, thickness=7):
        x, y, width, height = boundingBox
        x1, y1 = x+width, y+height
        cv.rectangle(image, boundingBox, (0, 255, 0), 1)
        # Top Left Line
        cv.line(image, (x,y), (x+length,y), (0,255,0), thickness)
        cv.line(image, (x, y), (x, y+length), (0, 255, 0), thickness)
        # Top Right Line
        cv.line(image, (x+width, y), (x +width- length, y), (0, 255, 0), thickness)
        cv.line(image, (x+width, y), (x+width, y + length), (0, 255, 0), thickness)
        # Bottom Left Line
        cv.line(image, (x, y+height), (x + length, y+height), (0, 255, 0), thickness)
        cv.line(image, (x, y+height), (x, y - length+height), (0, 255, 0), thickness)
        # Bottom Right Line
        cv.line(image, (x + width, y+height), (x + width - length, y+height), (0, 255, 0), thickness)
        cv.line(image, (x + width, y+height), (x + width, y - length+height), (0, 255, 0), thickness)
        return image



def main():
    cap = cv.VideoCapture('Videos/1.mp4')
    cTime = 0
    pTime = 0
    detector= FaceDetector()
    while True:
        success, image = cap.read()
        image,boundingBoxes=detector.findFaces(image)
        print(boundingBoxes)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(image, f'FPS:{int(fps)}', (50, 90), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv.imshow("Image", image)
        cv.waitKey(10)

if __name__ == '__main__':
    main()