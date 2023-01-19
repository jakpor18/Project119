import cv2
import time
import math

video = cv2.VideoCapture("footvolleyball.mp4")

#Load tracker
tracker = cv2.TrackerCSRT_create()
#Read first frame of the video
returned,img= video.read()
#Select the bounding box of the image
bbox= cv2.selectROI("tracking",img,False)
#Initalize the tracker pn the image and the bounding box
tracker.init(img,bbox)
print(bbox)
def drawbox(img,bbox):
    x,y,w,h=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img,"tracking",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)   
while True:
    check,img = video.read()   
    #Update the tracker on the image and the bounding box
    success,bbox= tracker.update(img)
    if success:
        drawbox(img,bbox)
    else:
        cv2.putText(img,"lost",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) 

    cv2.imshow("result",img)
            
    key = cv2.waitKey(25)

    if key == 32:
        print("Stopped!")
        break


video.release()
cv2.destroyALLwindows()



