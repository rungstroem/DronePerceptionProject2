import cv2
import numpy as np

def frameViewer():

    cap = cv2.VideoCapture('DJI_0199.mov')

    count = 0
    n = 0
    while(1):
        ret, frame = cap.read()
        if ret is not True:
            break
        
        if ret:
            if n < 10:
                cv2.imwrite('frame0{:d}.jpg'.format(n), frame)
            else:
                cv2.imwrite('frame{:d}.jpg'.format(n), frame)
            count += 50 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
            n += 1
        else:
            cap.release()
            break
    

    cap.release()

def main():
	frameViewer()
main()

