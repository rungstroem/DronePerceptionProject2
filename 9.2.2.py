#9.2.2, matching features between 2 frames and estimate the essential matrix

def match():

    image1 = cv2.imread("first_fram.jpg")
    image2 = cv2.imread("second_frame.jpg")

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_image1, None)
    kp2, des2 = sift.detectAndCompute(gray_image2, None)

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
            ransacReprojecThreshold)

    img3 = cv2.drawMatches(image1, kp1,
            image2, kp2,
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

    cv2.imwrite("first_two_frames_matching_features, img3)



main()