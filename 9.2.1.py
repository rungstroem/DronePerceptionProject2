#9.2.1, SIFT detection

import cv2
import numpy as np

def main():
    img = cv2.imread("the_image_you_need.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray)

    img=cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("output/ex01_image_with_sift_features.png", img)


main()






