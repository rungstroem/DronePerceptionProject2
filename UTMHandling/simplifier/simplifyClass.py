from math import pi, cos, sin, acos, atan2, sqrt
import matplotlib.pyplot as plt
from rdp import rdp

class simplifyClass:
	def maxWaypoints(points, fileName):
		x = [];
		y = [];
		z = [];
		hemi = [];
		zone = [];
		lett = [];
		xCorr = [];	# For plotting
		yCorr = [];	# For plotting
		
		fData = open(fileName, 'r');
		for line in fData :
			csv = line.split(',');
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
			hemi.append(csv[3]);
			zone.append(int(csv[4]));
			lett.append(csv[5]);
		fData.close();

		fCorr = open(fileName.replace(".csv", "")+"_maxPoints.csv", 'w', newline='');
		
		#Define number sample interval
		sample = int(len(x)/points);
		print(sample)
		i = 0;
		while(i<len(x)) :
			xCorr.append(x[i]);
			yCorr.append(y[i]);
			fCorr.write(str(x[i]));
			fCorr.write(',');
			fCorr.write(str(y[i]));
			fCorr.write(',');
			fCorr.write(str(z[i]));
			fCorr.write(',');
			fCorr.write(str(hemi[i]));
			fCorr.write(',');
			fCorr.write(str(zone[i]));
			fCorr.write(',');
			fCorr.write(lett[i]);
			#fCorr.write('\n');		#Line break is needed if last write is not a letter
			i += sample;

		if(i != len(x)) :
			fCorr.write(str(x[len(x)-1]));
			fCorr.write(',');
			fCorr.write(str(y[len(x)-1]));
			fCorr.write(',');
			fCorr.write(str(z[len(x)-1]));
			fCorr.write(',');
			fCorr.write(str(hemi[len(x)-1]));
			fCorr.write(',');
			fCorr.write(str(zone[len(x)-1]));
			fCorr.write(',');
			fCorr.write(lett[len(x)-1]);
			#fCorr.write('\n');		#Line break is needed if last write is not a letter
		fCorr.close();
		
		# Just plotting
		fig, axs = plt.subplots(2)
		fig.suptitle("Graph for UTM data vs max n waypoints");
		axs[0].scatter(x, y);
		axs[0].set_title("UTM data");
		axs[0].axis('equal')
		axs[1].scatter(xCorr, yCorr);
		axs[1].set_title("Only n waypoints");
		axs[1].axis('equal');
		plt.show()
	
	def minDeviation(fileName):
		x = [];
		y = [];
		z = [];
		xCorr = [];
		yCorr = [];
		
		fData = open(fileName, 'r');
		for line in fData:
			csv = line.split(',');
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
		fData.close();
		
		
	def douglasPeucker(fileName, epsilon):
		fData = open(fileName,'r');
		x = [];
		y = [];
		z = [];
		hemi = [];
		zone = [];
		lett = [];
		M = [];
		MCorr = [];
		for line in fData:
			csv = line.split(',')
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
			hemi.append(csv[3]);
			zone.append(int(csv[4]));
			lett.append(csv[5]);
			M.append([float(csv[0]),float(csv[1]),float(csv[2])]);
		fData.close();
		
		MCorr = rdp(M,epsilon);
		xCorr = [];
		yCorr = [];
		fCorr = open(fileName.replace(".csv", "")+"_RDPCorr.csv", 'w',newline='');
		for i in range(0,len(MCorr)):
			fCorr.write(str(MCorr[i][0]));
			fCorr.write(',');
			fCorr.write(str(MCorr[i][1]));
			fCorr.write(',');
			fCorr.write(str(MCorr[i][2]));
			fCorr.write(',');
			fCorr.write(str(hemi[i]));
			fCorr.write(',');
			fCorr.write(str(zone[i]));
			fCorr.write(',');
			fCorr.write(lett[i]);
			#fCorr.write('\n');		#Line break is needed if the last write is not a letter
			xCorr.append(MCorr[i][0]);
			yCorr.append(MCorr[i][1]);
		fCorr.close();
			
		# Just plotting
		fig, axs = plt.subplots(2)
		fig.suptitle("Graph for UTM data vs RDP-algorothm");
		axs[0].scatter(x, y);
		axs[0].set_title("UTM data");
		axs[0].axis('equal')
		axs[1].scatter(xCorr, yCorr);
		axs[1].set_title("RDP-algorithm");
		axs[1].axis('equal');
		plt.show()
