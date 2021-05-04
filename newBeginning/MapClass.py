import cv2
import numpy as np
import glob
import pangolin
import OpenGL.GL as gl
from display import Display3D
import collections
import g2opy as g2o


Feature = collections.namedtuple('Feature', 
        ['keypoint', 'descriptor', 'feature_id'])
Match = collections.namedtuple('Match', 
        ['featureid1', 'featureid2', 
            'keypoint1', 'keypoint2', 
            'descriptor1', 'descriptor2', 
            'distance', 'color'])
Match3D = collections.namedtuple('Match3D', 
        ['featureid1', 'featureid2', 
            'keypoint1', 'keypoint2', 
            'descriptor1', 'descriptor2', 
            'distance', 'color', 
            'point'])
MatchWithMap = collections.namedtuple('MatchWithMap', 
        ['featureid1', 'featureid2', 
            'imagecoord', 'mapcoord', 
            'descriptor1', 'descriptor2', 
            'distance'])





class Observation():
    def __init__(self, point_id, camera_id, image_coordinates):
        self.point_id = point_id
        self.camera_id = camera_id
        self.image_coordinates = image_coordinates

    def __repr__(self):
        return repr("Observation - point %d - camera %d (%f %f)" % (
            self.point_id,
            self.camera_id,
            self.image_coordinates[0],
            self.image_coordinates[1]))
        
class Map():
    def __init__(self):
        self.clean()

    def clean(self):
        self.next_id_to_use = 0
        self.points = []
        self.cameras = []
        self.observations = []
        self.camera_matrix = None

    def increment_id(self):
        t = self.next_id_to_use
        self.next_id_to_use += 1
        return t
    
    def add_camera(self, camera):
        camera.camera_id = self.increment_id()
        self.cameras.append(camera)
        return camera

    def add_point(self, point):
        point.point_id = self.increment_id()
        self.points.append(point)
        return point


    def calculate_reprojection_error_for_point(self, point, camera_pose, image_coord):
        temp = camera_pose @ point
        projected_image_coord = self.camera_matrix @ temp[0:3]
        projected_image_coord /= projected_image_coord[2]
        dx = projected_image_coord[0] - image_coord[0]
        dy = projected_image_coord[1] - image_coord[1]
        return dx**2 + dy**2


    def calculate_reprojection_error(self):
        camera_dict = {}
        for camera in self.cameras:
            camera_dict[camera.camera_id] = camera
        point_dict = {}
        for point in self.points:
            point_dict[point.point_id] = point
        total_error = 0
        for observation in self.observations:
            camera = camera_dict[observation.camera_id]
            point = point_dict[observation.point_id]
            point = np.array(point.point)
            point = np.hstack((point, 1))
            point = np.array([point]).T
            point_in_cam_coords = camera.pose() @ point
            t = self.camera_matrix @ point_in_cam_coords[0:3, :]
            t = t / t[2, 0]
            dx = t[0] - observation.image_coordinates[0]
            dy = t[1] - observation.image_coordinates[1]
            total_error += np.abs(dx*dx) + np.abs(dy*dy)

        print("calculated reprojection error")
        print("total error: %f" % total_error)


    def optimize_map(self, postfix = ""):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Define camera parameters
        print(self.camera_matrix)
        #focal_length = 1000
        focal_length = self.camera_matrix[0, 0]
        #principal_point = (320, 240)
        principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        baseline = 0
        cam = g2o.CameraParameters(focal_length, principal_point, baseline)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        camera_vertices = {}
        for camera in self.cameras:
            # Use the estimated pose of the second camera based on the 
            # essential matrix.
            pose = g2o.SE3Quat(camera.R, camera.t)

            # Set the poses that should be optimized.
            # Define their initial value to be the true pose
            # keep in mind that there is added noise to the observations afterwards.
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(camera.camera_id)
            v_se3.set_estimate(pose)
            v_se3.set_fixed(camera.fixed)
            optimizer.add_vertex(v_se3)
            camera_vertices[camera.camera_id] = v_se3
            #print("camera id: %d" % camera.camera_id)

        point_vertices = {}
        for point in self.points:
            # Add 3d location of point to the graph
            vp = g2o.VertexPointXYZ()
            vp.set_id(point.point_id)
            vp.set_marginalized(True)
            # Use positions of 3D points from the triangulation
            point_temp = np.array(point.point, dtype=np.float64)
            vp.set_estimate(point_temp)
            optimizer.add_vertex(vp)
            point_vertices[point.point_id]= vp


        for observation in self.observations:
            # Add edge from first camera to the point
            edge = g2o.EdgeProjectXYZ2UV()

            # 3D point
            edge.set_vertex(0, point_vertices[observation.point_id]) 
            # Pose of first camera
            edge.set_vertex(1, camera_vertices[observation.camera_id]) 
            
            edge.set_measurement(observation.image_coordinates)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))

        print('Performing full BA:')
        optimizer.initialize_optimization()
        optimizer.set_verbose(True)
        optimizer.optimize(40)
        optimizer.save("test.g2o");

        for idx, camera in enumerate(self.cameras):
            t = camera_vertices[camera.camera_id].estimate().translation()
            self.cameras[idx].t = t
            q = camera_vertices[camera.camera_id].estimate().rotation()
            self.cameras[idx].R = quarternion_to_rotation_matrix(q)

        for idx, point in enumerate(self.points):
            p = point_vertices[point.point_id].estimate()
            # It is important to copy the point estimates.
            # Otherwise I end up with some memory issues.
            # self.points[idx].point = p
            self.points[idx].point = np.copy(p)
