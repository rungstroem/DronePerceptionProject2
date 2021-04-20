#9.2.2, matching features between 2 frames and estimate the essential matrix
import cv2
import numpy as np

#def cameraMatrix():

    # width = 3840
    # height = 2160
    # focalLength = 2676.1051390718389
    # cx = -35.243952918157035
    # cy = -279.58562078697361
    # k1 = 0.0097935857180804498
    # k2 = -0.021794052829051412
    # k3 = 0.017776502734846815
    # p1 = 0.0046443590741258711
    # p2 = -0.0045664024579022498
    # date = ("2021-04-20T08:07:51Z")

    #K = [[focalLength, 0, cx],[0,focalLength,cy],[0,0,1]]




def match():

    focalLength = 2676.1051390718389
    cx = -35.243952918157035
    cy = -279.58562078697361
    cameraMatrix = [[focalLength, 0, cx],[0,focalLength,cy],[0,0,1]]

    image1 = cv2.imread("frame0.jpg")
    image2 = cv2.imread("frame50.jpg")

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp_image1, des1 = sift.detectAndCompute(gray_image1, None)
    kp_image2, des2 = sift.detectAndCompute(gray_image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)

    points1_temp = []
    points2_temp = []
    match_indices_temp = []

    for idx, m in enumerate(matches):
        points1_temp.append(kp_image1[m.queryIdx].pt)
        points2_temp.append(kp_image2[m.trainIdx].pt)
        match_indices_temp.append(idx)

    points1 = np.float32(points1_temp)
    points2 = np.float32(points2_temp)
    match_indices = np.int32(match_indices_temp)
    ransacReprojecThreshold = 1
    confidence = 0.99

    # Finding the essential matrix
    essentialMatrix, mask = cv2.findEssentialMat(
            points1,
            points2,
            cameraMatrix,
            cv2.FM_RANSAC,
            confidence,
            ransacReprojecThreshold,
            None)

    img3 = cv2.drawMatches(image1, kp_image1,
            image2, kp_image2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    match_indices = match_indices[mask.ravel()==1]
    filtered_matches = []
    for index in match_indices:
        m = matches[index]
        filtered_matches.append(m)


    img3 = cv2.drawMatches(img_image1, kp_image1,
            img_image2, kp_image2,
            filtered_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite("first_two_frames_matching_features", img3)



match()