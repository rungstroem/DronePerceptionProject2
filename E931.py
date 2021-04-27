import numpy as np;
import cv2
import OpenGL.GL as gl
import pangolin
import sys

#sys.path.append('.');
#from visual_slam import FrameGenerator, Map, VisualSlam, Observation

def loadImage(name):
	return cv2.imread("../IMG/"+name);

def extractSift(img):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
	sift = cv2.SIFT_create();
	kp, des = sift.detectAndCompute(imgGray, None);
	return(kp, des);

def matcher(des1, des2):
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True);
	matches = bf.match(des1, des2);
	return matches;

def find_fundamental_matrix(p1List, p2List):
	confidence = 0.95;
	iterations = 10;
	F, mask = cv2.findFundamentalMat(p1List, p2List, cv2.FM_RANSAC, 1, confidence, iterations);
	return(F, mask);

def draw_matches(img1, img2, kp1, kp2, matches, name):
	global show;
	img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2);
	if(show == True):
		cv2.imshow(name, img);
		cv2.waitKey(0);

def find_essential_matrix(cameraMatrix, p1List, p2List):
	confidence = 0.95;
	E, mask = cv2.findEssentialMat(p1List, p2List, cameraMatrix, cv2.FM_RANSAC, confidence, 1)
	return(E, mask);

def get_number_of_inliers(mask):
	inliers = 0;
	for i in range(len(mask.ravel())):
		if(mask.ravel()[i] == 1):
			inliers += 1;
	return inliers;

def decompose_essential_matrix(E):
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]]);
	U, Sigma, V = np.linalg.svd(E, full_matrices=True);
	V = V.T;	# V from svd is already transposed
	t1 = U[:,2];
	t2 = -U[:,2];
	R1 = U @ W.T @ V.T;
	R2 = U @ W @ V.T;
	return(R1, R2, t1, t2);

def filter_matches(matches, mask):
	index = [];
	for i,m in enumerate(matches):
		index.append(i);
	index = np.int32(index);
	index = index[mask.ravel()==1];
	matchesFiltered = [];
	for i in index:
		matchesFiltered.append(matches[i]);
	
	return matchesFiltered;

def extract_matched_points(matches, kp1, kp2):
	points1 = [];
	points2 = [];
	for i,m in enumerate(matches):
		print(i);
		points1.append(kp1[m.queryIdx].pt);
		points2.append(kp2[m.trainIdx].pt);
	points1F = np.float32(points1);
	points2F = np.float32(points2);
	return (points1F, points2F);

def get_epipolar_line(p1List, p2List):
	F = find_fundamental_matrix(p1List, p2List);
	return F*p1List[1];

def est_camera_pose(E, p1F, p2F, K):
	point, R, t, mask = cv2.recoverPose(E, p1F, p2F, K);
	return(point, R, t, mask);

def get_camera_poses(E, p1F, p2F, K):
	# calculate rotation and translation from one pose to the other
	point, R, t, mask = cv2.recoverPose(E, p1F, p2F, K);
	# Create Null mat for camera pose used as starting point
	mat1 = np.hstack((np.eye(3,3), np.zeros((3,1))));
	# Create mat with rotation and translation from starting point (Null mat)
	mat2 = np.hstack((R,t));
	# Create the poses by multiplying with camera matrix, ie. the intrinsic parameters for camera
	pose1 = K @ mat1;
	pose2 = K @ mat2;
	return(pose1, pose2);

def reconstruct_3d(pose1, pose2, p1F, p2F):
	# Get 3D points from triangulation
	points3D = cv2.triangulatePoints(pose1, pose2, p1F.T, p2F.T);
	# Scale points with z
	points3D /= points3D[3,:];
	return points3D;

def visualize_3D(p3D):
	pangolin.CreateWindowAndBind('main', 640, 480);
	gl.glEnable(gl.GL_DEPTH_TEST);
	scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(640,480,420,420,320,240,0.2,100),
			pangolin.ModelViewLookAt(-2,2, -2, 0,0,0, pangolin.AxisDirection.AxisY));
	handler = pangolin.Handler3D(scam);
	
	dcam = pangolin.CreateDisplay();
	dcam.SetBounds(0.0,1.0,0.0,1.0,-640.0/480.0);
	dcam.SetHandler(handler);

	while not pangolin.ShouldQuit():
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
		gl.glClearColor(1.0,1.0,1.0,1.0);
		dcam.Activate(scam);

		#gl.glPointSize(5);
		#gl.glColor3f(0.0,0.0,1.0);
		#pangolin.DrawPoints(p3D);
		
		# Attempt to draw camera
		gl.glLineWidth(2);
		gl.glColor3f(1.0,0.0,0.0);
		#pose = np.identity(4);
		pose = (np.hstack((pose1.T, np.zeros((4,1))))).T;
		print(pose)
		#pangolin.DrawCamera(pose, 0.5,0.75,0.8);

		#pangolin.FinishFrame();

def main():
	global show;
	show = False;
	# The camera matrix contains focal length and ...
	K = np.array([[704, 0, 637],[0, 704, 376],[0, 0, 1]]);	#Needed to find essential matrix
	
	img1 = loadImage("frame0.jpg");
	img2 = loadImage("frame50.jpg");
	
	# Extract keypoints and descriptors from both images
	kp1, des1 = extractSift(img1);
	kp2, des2 = extractSift(img2);
	# Find points that are visible in both images - matching points
	matches = matcher(des1, des2);
	#draw_matches(img1, img2, kp1, kp2, matches, "Matches with outliers");
	
	# Find coordinates of points in both images - extract from keypoints
	(points1F, points2F) = extract_matched_points(matches, kp1, kp2);
	
	# Find essential matrix from matched coordinates
	(E, maskE) = find_essential_matrix(K, points1F, points2F);
	
	# Filter coordinates with essential matrix
	filteredMatches = filter_matches(matches, maskE);

	# Find new coordinates with the filtered list of matches
	(p1FFiltered, p2FFiltered) = extract_matched_points(filteredMatches, kp1, kp2);
	print(p1FFiltered, p2FFiltered);
	# Find the epipolar line with the fundamental matrix F
	L2 = get_epipolar_line(p1Filtered, p2Filtered);
	
	# Decompose essential matrix into 2XRotation matrix and 2XTranslation vector
	(R1, R2, t1, t2) = decompose_essential_matrix(E);
	
	# Estimate camera pose
	(pose1, pose2) = get_camera_poses(E, p1FFiltered, p2FFiltered, K);
	
	# extract 3D points
	D3Points = reconstruct_3d(pose1, pose2, p1FFiltered, p2FFiltered);


main();
