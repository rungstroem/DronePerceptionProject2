import cv2
import numpy as np
# import MarkerLocator.MarkerTracker as MarkerTracker

def frameViewer():
    #filename = './videos/video_with_n_fold_markers.mov'
    cap = cv2.VideoCapture('DJI_0199.mov')

    count = 0

    #tracker = MarkerTracker.MarkerTracker(order=4, kernel_size=25, scale_factor=0.1)

    #tracker.track_marker_with_missing_black_leg = False
    while(1):
        ret, frame = cap.read()
        if ret is not True:
            break

        # frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); 
        # imgScale = cv2.resize(frameGray, None, fx=0.4, fy=0.4);       
	
        # markerPose = tracker.locate_marker(imgScale)
        # magnitude = np.sqrt(tracker.frame_sum_squared)
        
        if ret:
            cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            count += 50 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break
        
	# Process frame
        #cv2.imshow('frame',magnitude/5)

        # Deal with key presses
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite("output/ex00_stillimage.png", frame)

    cap.release()

def main():
	frameViewer()
main()

# import cv2

# cap = cv2.VideoCapture('XYZ.avi')
# count = 0

# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret:
#         cv2.imwrite('frame{:d}.jpg'.format(count), frame)
#         count += 30 # i.e. at 30 fps, this advances one second
#         cap.set(1, count)
#     else:
#         cap.release()
#         break