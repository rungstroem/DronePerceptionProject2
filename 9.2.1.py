#9.2.1, SIFT detection

import cv2
import numpy as np

def main():
    img = cv2.imread("test.jpeg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('out_image.png', kp_img)


main()





