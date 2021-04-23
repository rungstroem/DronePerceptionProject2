from math import pi, cos, sin, acos, atan2, sqrt
import matplotlib.pyplot as plt

class outlierClass:
	def __init__(self, freq, vel, fileName):
		dt = freq;		# Hz
		velocity = vel;	# m/s
		x = [];
		y = [];
		z = [];
		hemi = [];
		zone = [];
		lett = [];
		xCorr = [];
		yCorr = [];
		
		fData = open(fileName, 'r');
		for line in fData :
			csv = line.split(',');
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
			hemi.append(csv[3]);
			zone.append(int(csv[4]));
			lett.append(csv[5]);
		fData.close;

		fCorr = open(fileName.replace(".csv", "")+"_corrected.csv", 'w',newline='');

		cOld = sqrt(x[0]**2+y[0]**2);
		for i in range(0,len(x)):
			cNew = sqrt(x[i]**2 + y[i]**2);
			if( (cNew - cOld) < velocity*dt ):
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
				#fCorr.write('\n');		#Line break is needed if last write is not a letter.
				xCorr.append(x[i]);
				yCorr.append(y[i]);
			cOld = cNew;
		fCorr.close();

		fig, axs = plt.subplots(nrows=2)
		fig.suptitle("Graph for UTM data before and after removing outliers");
		axs[0].scatter(x, y);
		axs[0].set_title("Uncorrected UTM data");
		axs[0].axis('equal')
		axs[0].set_xlabel('N');
		axs[0].set_ylabel('E');
		axs[1].scatter(xCorr, yCorr);
		axs[1].set_title("Corrected UTM data");
		axs[1].axis('equal');
		axs[1].set_xlabel('N');
		axs[1].set_ylabel('E');
		plt.show()

