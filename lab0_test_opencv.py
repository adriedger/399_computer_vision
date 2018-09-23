import numpy as np
import cv2

# runs through two loops. each loop will give a video feed from webcam and when you hit 'q' you will
# take a picture. At the end it tries to match the two pictures together. I haven't included
# any warp to the pictures, but it tries to connect features of each picture.


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, im1 = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imshow('image',im1)
cv2.waitKey(0)
while(True):
    # Capture frame-by-frame
    ret, im2 = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# show captured image
cv2.imshow('image',im2)
cv2.waitKey(0)
# save image
cv2.imwrite('testfile.tif', im1)

# read saved image
reloaded = cv2.imread('testfile.tif', cv2.IMREAD_GRAYSCALE)
# apply heavy gaussian blur
blurred = cv2.GaussianBlur(reloaded,(99,99),0)
cv2.imshow('image',blurred)
cv2.waitKey(0)

