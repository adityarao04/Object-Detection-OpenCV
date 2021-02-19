import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500
while True:
    _,img = cam.read()
    text = "Normal"
    img = imutils.resize(img,width=500)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg,(21,21),0)

    # Setting up first Frame
    if firstFrame is None:
            firstFrame = gaussianImg
            continue
    # after geeting first frame we need to find the difference b/w ff and gaussianImg
    imgDiff = cv2.absdiff(firstFrame,gaussianImg)
    # Now we find threshold
    # cv2.threshold(source image,threshold,value to set,thresthold model)
    threshImg = cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    # Now we apply dilation:Dilation adds pixels to the boundaries of objects in an image
    # dilate() takes the following values:
    # 1. image: It is a required parameter and an original image on which we need to perform dilation.
    # 2.kernel: It is the required parameter is the matrix with which the image is convolved.
    # 3.dst: It is the output image of the same size and type as image src.
    # 4.anchor: It is a variable of type integer representing the anchor point and its default value Point is (-1, -1) which means that the anchor is at the kernel center.
    # 5. borderType: It depicts what kind of border to be added. It is defined by flags like cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, etc.
    # 6.iterations: It is an optional parameter that takes several iterations.
    # 7.borderValue: It is border value in case of a constant border.          
    threshImg = cv2.dilate(threshImg,None,iterations=2)
    # Contours will help us detect edges and in sahe analysis 
    cnts = cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Now we detect movement
    for c in cnts:
             if cv2.contourArea(c) < area:
                    continue
             (x,y,w,h) = cv2.boundingRect(c)
             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
             text = "Moving Object Detected"
             
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
    # Displaying objects
    cv2.imshow("cameraFeed",img)
    # cv2.imshow("Threshold Image",threshImg)

    key = cv2.waitKey(1)&0xFF
    # q to quit window
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()