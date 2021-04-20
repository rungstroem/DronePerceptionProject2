#!/usr/bin/env python3
f = open("positions2.csv","r")
ff = open("UTM_positions2.csv","w")
# import utmconv class
from utm import utmconv
from math import pi, cos, acos, sin, atan2, sqrt


for line in f:

	csv = line.split(",")
    
# Geodetic coordinates
	lat1 =  float(float(csv[0]));
	lon1 = float(float(csv[1]))
	z = float(float(csv[2]))

# Instantiate utmconv class
	uc = utmconv()
	(hemisphere, zone,letter, e, n) = uc.geodetic_to_utm(lat1,lon1)
	
# Save to file
	ff.write(str(e))
	ff.write(",")
	ff.write(str(n))
	ff.write(",")
	ff.write(str(z))
	ff.write(",")
	ff.write(str(hemisphere))
	ff.write(",")
	ff.write(str(zone))
	ff.write(",")
	ff.write(str(letter))
	ff.write("\n")

f.close()
ff.close()
