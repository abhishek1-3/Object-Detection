from skimage.io import imshow
from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model=YOLO('../yoloweights/yolov8l.pt')
image=cv2.imread("images/13.jpg")
img=cv2.resize(image,(1080,720))
result=model(img)
pre=result[0].plot()
cv2.imshow("Detection",pre)
cv2.waitKey(0)
model=YOLO('../yoloweights/yolov8n.pt')
image1=cv2.imread("images/13.jpg")
img1=cv2.resize(image1,(1080,720))
result=model(img1)
pre=result[0].plot()
cv2.imshow("Detection",pre)
cv2.waitKey(0)




