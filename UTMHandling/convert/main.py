import csv
from utm import utmconv
from math import pi,cos,acos,sin,atan2,sqrt

outFile = open("./UTMCoordinates.csv" ,'w');

with open("../../data/FlightRecord.csv", newline='', encoding="windows-1252") as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	header = next(reader,None);	# This skips the header of the log file
	#for i in range(len(header)):
		#print(i, "  ", header[i]);
	for row in reader:
		# Geodetic coordinates
		lat = float(row[12]);
		lon = float(row[13]);

		# Instrantiate utmconv class
		uc = utmconv();
		(hemisphere, zone, letter, eastings, northings) = uc.geodetic_to_utm(lat,lon);

		# Save UTM to file
		outFile.write(str(eastings));
		outFile.write(',');
		outFile.write(str(northings));
		outFile.write(',');
		outFile.write(str(zone));
		outFile.write(',');
		outFile.write(str(letter));
		outFile.write("\n");

outFile.close();
