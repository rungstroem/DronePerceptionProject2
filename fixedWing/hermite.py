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
		# http://cubic.org/docs/hermite.htm
		# http://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation%20on%20a%20single%20interval
		p = []
		for t in range(steps):
			s = t/(steps * 1.0) # scale s to go from 0 to 1

			# # calculate basis function
			h1 = 2*s**3 - 3*s**2 + 1
			h2 = -2*s**3 + 3*s**2
			h3 = s**3 - 2*s**2 + s
			h4 = s**3 - s**2

			# multiply and sum functions together to build the interpolated point along the curve
			v1 = self.v2d_scalar_mul(p1,h1)
			v2 = self.v2d_scalar_mul(p2,h2)
			v3 = self.v2d_scalar_mul(t1,h3)
			v4 = self.v2d_scalar_mul(t2,h4)

			p.append(self.v2d_add(self.v2d_add(self.v2d_add(v1,v2),v3),v4))
		return p

chs = cubic_hermite_spline()

p1 = [0.0, 0.0]
t1 = [0.0, 5.0]
p2 = [2.0, 0.0]
t2 = [0.0, 5.0]
steps = 50

rte = chs.goto_wpt (p1,t1,p2,t2,steps)

print (rte)

# plot route
rteT = list(zip(*rte))
rte_plt = plot(rteT[0],rteT[1],'blue')

title ('Route')
axis('equal')
xlabel('Easting [m]')
ylabel('Northing [m]')
plt.savefig ('route_plan.png')
ion() # turn interaction mode on
show()
ioff()

