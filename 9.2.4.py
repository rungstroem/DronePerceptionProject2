#9.2.4 Recovering the motion

def match():

    cameraMatrix = np.array([[2676.1051390718389,   0.,         -35.243952918157035 ],
                            [  0.,         2676.1051390718389, -279.58562078697361 ],
                            [0., 0., 1.]])


    image1 = cv2.imread("Test_1.jpg")
    image2 = cv2.imread("Test_2.jpg")

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
        points1_temp.append(kp1[m.queryIdx].pt)
        points2_temp.append(kp2[m.trainIdx].pt)
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


    match_indices = match_indices[mask.ravel()==1]
    filtered_matches = []
    for index in match_indices:
        m = matches[index]
        filtered_matches.append(m)




    retval, R, t, mask = cv2.recoverPose(essentialMatrix, points1, points2, cameraMatrix)
    print(retval)
    print(R)
    print(t)
    tx = np.array([[0, -t[2,0], t[1,0]], [t[2,0], 0, -t[0,0]], [-t[1,0], t[0,0], 0]])

    estimated_essential_matrix = tx @ R * -0.72
    print(estimated_essential_matrix)
    print(essentialMatrix)



match()