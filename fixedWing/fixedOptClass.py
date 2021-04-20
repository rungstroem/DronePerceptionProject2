#!/usr/bin/env python3
# implementation of a cubic hermite spline

# load modules
from math import pi
import matplotlib.pyplot as plt
from pylab import *

class cubic_hermite_spline():
	def __init__(self):
		pass

	def v2d_scalar_mul (self, v, f):
		return [v[0]*f, v[1]*f]

	def v2d_add (self, v1, v2):
		return [v1[0]+v2[0], v1[1]+v2[1]]

	def goto_wpt (self, p1, t1, p2, t2, steps):
		p = []
		for t in range(steps):
			s = t/(steps * 1.0) # scale s to go from 0 to 1

			# # calculate basis function
			h1 = 2*s**3 - 3*s**2 + 1
			h2 = -2*s**3 + 3*s**2
			h3 = s**3 - 2*s**2 + s
			h4 = s**3 - s**2
			v1 = self.v2d_scalar_mul(p1,h1)
			v2 = self.v2d_scalar_mul(p2,h2)
			v3 = self.v2d_scalar_mul(t1,h3)
			v4 = self.v2d_scalar_mul(t2,h4)

			p.append(self.v2d_add(self.v2d_add(self.v2d_add(v1,v2),v3),v4))
		return p

class fixedOptClass :
	def __init__(self):
		pass;
	
	def spline(self,a,p2,p1):
		return a*(p2-p1);
		
	def generateRoute(self,fileName, intPoints) :
		x = [];
		y = [];
		z = [];
		hemi = [];
		zone = [];
		lett = [];
		xCorr = [];
		yCorr = [];
		fData = open(fileName, 'r');
		for line in fData:
			csv = line.split(',');
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
			hemi.append(csv[3]);
			zone.append(float(csv[4]));
			lett.append(csv[5]);
		fData.close;
		
		# Init class
		chs = cubic_hermite_spline();
		route = []
		for i in range(0,len(x)-2):
			p1 = (x[i],y[i]);
			p2 = (x[i+1],y[i+1]);
			t1 = ( self.spline(0.9,x[i+1],x[i]), self.spline(0.9,y[i+1],y[i]) );
			t2 = ( self.spline(0.9,x[i+2],x[i+1]), self.spline(0.9,y[i+2], y[i+1]) );
			route = chs.goto_wpt(p1,t1,p2,t2, intPoints);
			xCorr.append(x[i]);
			yCorr.append(y[i]);
			
			for j in range(0,intPoints):
				xCorr.append(route[j][0]);
				yCorr.append(route[j][1]);

		plt.scatter(xCorr,yCorr);
		plt.axis('equal');
		plt.title("Interpolated points");
		plt.xlabel("N");
		plt.ylabel("E");
		plt.show();
		
