import numpy as np;
from numpy import random
import cv2
import OpenGL.GL as gl
import pangolin
import matplotlib.pyplot as plt
#import sys

#sys.path.append('.');
#from visual_slam import Map #FrameGenerator, Map, VisualSlam, Observation

def loadImage(name):
	return cv2.imread("../../IMG/"+name);

def scale_img(img):
	sf = 30;
	(h,w) = img.shape[:2];
	dim = (int((w/100)*sf), int((h/100)*sf));

	return cv2.resize(img, dim, cv2.INTER_AREA);

def scale_camera_mat(K):
	scalingFactor = 0.3;
	Kmat = K*scalingFactor;
	Kmat[2][2] = 1;
	return Kmat;

################################################################################################
# Map class
class Map():
	def __init__():
		self.id = 0;
		self.points = [];
		self.cameras = [];
		self.observations = [];



################################################################################################
# Exercises
# Exercise 9.2.1 - choose feature detector
def extractSift(img):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
	sift = cv2.SIFT_create();
	kp, des = sift.detectAndCompute(imgGray, None);
	return(kp, des);

# Exercise 9.2.2 - Match features
def matcher(des1, des2):
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True);
	matches = bf.match(des1, des2);
	return matches;

# Exercise 9.2.2 - Estimate essential matrix
def find_essential_matrix(cameraMatrix, p1List, p2List):
	confidence = 0.95;
	E, mask = cv2.findEssentialMat(p1List, p2List, cameraMatrix, cv2.FM_RANSAC, confidence, 1)
	return(E, mask);

# Exercise 9.2.3 - Get epipolar line between points + statistics
def find_fundamental_matrix(p1List, p2List):
	confidence = 0.95;
	iterations = 10;
	F, mask = cv2.findFundamentalMat(p1List, p2List, cv2.FM_RANSAC, 1, confidence, iterations);
	return(F, mask);

def get_epipolar_lines(p1List, p2List, E, img1, img2):
	F, mask = find_fundamental_matrix(p1List, p2List);
	#lines1 = cv2.computeCorrespondEpilines(p2List.reshape(-1,1,2), 2, F);
	#lines1 = lines1.reshape(-1,3);
	#lines2 = cv2.computeCorrespondEpilines(p1List.reshape(-1,1,2), 2, F);
	#lines2 = lines2.reshape(-1,3);
	homogenCoord1 = to_homogen_coordinates(p1List);	
	homogenCoord2 = to_homogen_coordinates(p2List);
	
	
	lines1 = [];	# Epilines in image1
	for point in zip(homogenCoord2):
		lines1.append((point @ F).flatten());
	lines2 = [];	# Epilines in image2
	for point in zip(homogenCoord1):
		lines2.append((point @ F).flatten());
	
	draw_epi_lines(img1, img2, lines1, lines2, p1List, p2List);
	
	dList1 = calc_dist_to_epiline(homogenCoord2, lines1);
	dList2 = calc_dist_to_epiline(homogenCoord1, lines2);
	
	stat1 = statistics(dList1);
	print(stat1);
	stat2 = statistics(dList2);	
	print(stat2);
	draw_histogram(dList1, dList2, stat1, stat2);
	
def calc_dist_to_epiline(pList, lines):
	d = [];
	for point,line in zip(pList, lines):
		temp = np.sqrt( ((line @ point)/( np.sqrt(line[0]**2 + line[1]**2) ) )**2 );
		d.append(temp);
	return(d);

def statistics(dList):
	mean = sum(dList)/len(dList);
	var = 0;
	for i in range(len(dList)):
		var += (dList[i]-mean)**2;
	var /= len(dList)-1;
	std = var**0.5;
	return (mean, var, std);

# Exercise 9.2.4 - Estimate relative motion of camera between 2 frames
def est_camera_pose(E, p1F, p2F, K):
	point, R, t, mask = cv2.recoverPose(E, p1F, p2F, K);
	return(point, R, t, mask);

# Exercise 9.2.5 - Estimate 3d positions of the corresponding image points by triangulation
def get_camera_poses(E, p1F, p2F, K):
	# calculate rotation and translation from one pose to the other
	point, R, t, mask = cv2.recoverPose(E, p1F, p2F, K);
	# Create Null mat for camera pose used as starting point
	mat1 = np.hstack((np.eye(3,3), np.zeros((3,1))));
	# Create mat with rotation and translation from starting point (Null mat)
	mat2 = np.hstack((R,t));
	# Create the poses by multiplying with camera matrix, ie. the intrinsic parameters for camera
	pose1 = mat1;
	pose2 = mat2;
	return(pose1, pose2);

def reconstruct_3d(pose1, pose2, p1F, p2F, K):
	proj1 = to_projection_matrix(pose1, K);
	proj2 = to_projection_matrix(pose2, K);
	# Get 3D points from triangulation
	points4D = cv2.triangulatePoints(proj1, proj2, p1F.T, p2F.T);
	# Scale points with 4'th axis
	points4D /= points4D[3,:];
	points3D = points4D[:3,:].T;
	return (points3D);

# Exercise 9.2.7 - Calculate reprojection error
def calc_reprojection(points3D, proj, points):
	rePoints = [];
	for i in range(len(points3D)):
		point4D = np.hstack((points3D[i], 1));
		
		temp = point4D @ proj.T;
		temp /= temp[2];
		rePoints.append(temp);
	
	reProjError = [];
	errSum = 0;
	for i in range(len(rePoints)):
		temp = np.sqrt( (points[i][0]-rePoints[i][0])**2 + (points[i][1]-rePoints[i][1])**2 );
		reProjError.append(temp);
		errSum += temp;
	print(errSum);

##################################################################################################
# Helper functions
def decompose_essential_matrix(E):
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]]);
	U, Sigma, V = np.linalg.svd(E, full_matrices=True);
	V = V.T;	# V from svd is already transposed
	t1 = U[:,2];
	t2 = -U[:,2];
	R1 = U @ W.T @ V.T;
	R2 = U @ W @ V.T;
	return(R1, R2, t1, t2);

def to_homogen_coordinates(pList):
	homogenCoord = [];
	for i in range(len(pList)):
		temp = np.array([pList[i][0], pList[i][1], 1]);
		homogenCoord.append(temp);
	return homogenCoord;

def to_projection_matrix(pose, K):
	return(K @ pose);

################################################################################################
# Filter functions
def get_number_of_inliers(mask):
	inliers = 0;
	for i in range(len(mask.ravel())):
		if(mask.ravel()[i] == 1):
			inliers += 1;
	return inliers;

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
		points1.append(kp1[m.queryIdx].pt);
		points2.append(kp2[m.trainIdx].pt);
	points1F = np.float32(points1);
	points2F = np.float32(points2);
	return (points1F, points2F);

################################################################################################
# Drawing functions
def draw_matches(img1, img2, kp1, kp2, matches, name):
	global show;
	img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2);
	if(show == True):
		cv2.imshow(name, img);
		cv2.waitKey(0);

def draw_epi_lines(img1, img2, lines1, lines2, pList1, pList2):
	img5, img6 = draw_epi_lines_helper(img1, img2, lines1, pList1, pList2);
	img3, img4 = draw_epi_lines_helper(img2, img1, lines2, pList2, pList1);
	if(show == True):
		cv2.imshow("EpilinesImg1", img5);
		cv2.imshow("EpilinesImg2", img3);
		cv2.waitKey(0);
	
def draw_epi_lines_helper(img1, img2, lines, pList1, pList2):
	r,c = img1.shape[:2];
	for r, pt1, pt2 in zip(lines, pList1, pList2):
		color = tuple(np.random.randint(0,255,3).tolist());
		x0,y0 = map(int, [0,-r[2]/r[1]]);
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]]);
		img1 = cv2.line(img1, (x0,y0),(x1,y1), color ,1);
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt1),5,color,-1);
	return(img1, img2);

def draw_histogram(dList1, dList2, stat1, stat2):
	fig, ax = plt.subplots(2);
	ax[0].hist(dList1, density=False, bins=30, color="green");
	ax[0].set_title("Histogram for distance between epiline in I1 and points in I2");
	ax[0].set_xlabel("Distances");
	#ax[0].set_ylabel("Consentration of lines");
	ax[0].axvline(stat1[0], color='k', linestyle='dashed', linewidth=1)
	ax[0].text(stat1[0]*1.021, 50, 'Mean: {:.2f}'.format(stat1[0]))
	
	ax[1].hist(dList2, density=False, bins=30, color="red");
	ax[1].set_title("Histogram for distance between epiline in I2 and points in I1");
	ax[1].set_xlabel("Distances");
	#ax[1].set_ylabel("Consentration of lines");
	ax[1].axvline(stat2[0], color='k', linestyle='dashed', linewidth=1)
	ax[1].text(stat2[0]*1.02, 50, 'Mean: {:.2f}'.format(stat2[0]))
	
	plt.tight_layout();
	plt.show();
	#plt.savefig("./histograms.png");

def visualize_3D(p3D, pose1=None, pose2=None):
	w = 640;
	h = 480;
	pangolin.CreateWindowAndBind('main', w, h);
	gl.glEnable(gl.GL_DEPTH_TEST);
	scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 100),
			pangolin.ModelViewLookAt(-2,2, -2, 0,0,0, pangolin.AxisDirection.AxisY));
	handler = pangolin.Handler3D(scam);
	
	dcam = pangolin.CreateDisplay();
	dcam.SetBounds(0.0,1.0,0.0,1.0, w/h);
	dcam.SetHandler(handler);

	while not pangolin.ShouldQuit():
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT);
		gl.glClearColor(0.0,0.0,0.0,0.0);
		dcam.Activate(scam);
	
		# Draw points
		gl.glPointSize(5);
		gl.glColor3f(0.0,0.0,1.0);
		pangolin.DrawPoints(p3D);
		
		# Draw camera
		gl.glLineWidth(3);
		gl.glColor3f(1.0,1.0,1.0);
		if(pose1 is not None):
			pose1 = (np.hstack((pose1.T, np.zeros((4,1))))).T;
			pose1[3][3] = 1;
			pangolin.DrawCamera(pose1, 0.5,0.75,0.8);
		
		if(pose2 is not None):
			pose2 = (np.hstack((pose2.T, np.zeros((4,1))))).T;
			pose2[3][3] = 1;
			pangolin.DrawCamera(pose2, 0.5,0.75,0.8);

		pangolin.FinishFrame();

def main():
	global show;
	show = True;
	# The camera matrix contains focal length and ...
	K = np.array([[2676, 0, (3840/2-35.24)],[0, 2676, (2160/2-279)],[0, 0, 1]]);	#Needed to find essential matrix
	
	img1 = loadImage("frame00.jpg");
	img2 = loadImage("frame01.jpg");

	img1 = scale_img(img1);
	img2 = scale_img(img2);

	K = scale_camera_mat(K);

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
	#draw_matches(img1, img2, kp1, kp2, filteredMatches, "Matches filtered");

	# Find new coordinates with the filtered list of matches
	(p1FFiltered, p2FFiltered) = extract_matched_points(filteredMatches, kp1, kp2);
	
	# To homogenious coordinates - ie. [u,v,1]^T
	# homogenCoord1 = to_homogen_coordinates(p1FFiltered);

	# Find the epipolar line with the fundamental matrix F
	get_epipolar_lines(p1FFiltered, p2FFiltered, E, img1, img2);
	#get_epipolar_lines(points1F, points2F, E, img1, img2);
	
	# Decompose essential matrix into 2XRotation matrix and 2XTranslation vector
	(R1, R2, t1, t2) = decompose_essential_matrix(E);
	
	# Estimate camera pose
	(pose1, pose2) = get_camera_poses(E, p1FFiltered, p2FFiltered, K);
	
	# extract 3D points
	D3Points = reconstruct_3d(pose1, pose2, p1FFiltered, p2FFiltered, K);

	#Projection matrices
	proj1 = to_projection_matrix(pose1,K);
	proj2 = to_projection_matrix(pose2,K);
	# Reprojection
	calc_reprojection(D3Points, proj1, p1FFiltered);

	# Plots points and camera poses in 3D environment - My work not the teachers...
	visualize_3D(D3Points, pose1, pose2);
	#visualize_3D(D3Points);
main();
