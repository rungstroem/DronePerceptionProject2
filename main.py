import numpy as np
import cv2
import OpenGL.GL as gl
import pangolin
import sys

sys.path.append('.');
from visual_slam import FrameGenerator, Map, VisualSlam, Observation


def main():
	vs = VisualSlam();
	mp = Map();
	ob = Observation();
	fg = FrameGenerator();


main();
