import csv
from math import pi,cos,acos,sin,atan2,sqrt
import matplotlib.pyplot as plt

eastings = [];
northings = [];

with open("./UTMCoordinates.csv", newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	#header = next(reader,None);	# This skips the header of the log file
	#for i in range(len(header)):
		#print(i, "  ", header[i]);
	for row in reader:
		# Geodetic coordinates
		eastings.append(float(row[0]));
		northings.append(float(row[1]));

plt.plot(eastings, northings);
plt.show();
