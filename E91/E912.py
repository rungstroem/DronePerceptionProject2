import cv2
import numpy as np

def frameViewer():

    cap = cv2.VideoCapture('DJI_0199.mov')

    count = 0

    while(1):
        ret, frame = cap.read()
        if ret is not True:
            break
        
        if ret:
            cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            count += 50 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break
    

    cap.release()

def main():
	frameViewer()
main()

