#!/usr/bin/env python3
import json

class routePlan:
	def route(inFile, outFile, cSpeed):
		x = [];
		y = [];
		z = [];
		fData = open(inFile, 'r');
		for line in fData :
			csv = line.split(',');
			x.append(float(csv[0]));
			y.append(float(csv[1]));
			z.append(float(csv[2]));
		fData.close();
		
		# Boiler plate code for creating a route plan
		plan = {};
		geoFence = {};
		plan['fileType'] = 'Plan';
		geoFence['polygon'] = [];
		geoFence['version'] = 1;
		plan['geoFence'] = geoFence;
		plan['groundStation'] = 'QGroundControl';
		items = [];
		
		# add datapoints
		item = {};
		item['autoContinue'] = True;
		item['command'] = 22;
		item['doJumpId'] = 1;
		item['frame'] = 3;
		item['params'] = [0,0,0,0,x[0],y[0],z[0]];
		item['type'] = 'SimpleItem';
		items.append (item);
		
		for i in range(1,len(x)) :
			item = {};
			item['autoContinue'] = True;
			item['command'] = 16;
			item['doJumpId'] = 2;
			item['frame'] = 3;
			item['params'] = [0,0,0,0,x[i],y[i],z[i]];
			item['type'] = 'SimpleItem';
			items.append (item);
		
		# Establish mission	
		mission = {}
		mission['cruiseSpeed'] = cSpeed;
		mission['firmwareType'] = 3;
		mission['hoverSpeed'] = 5;
		mission['items'] = items;
		mission['plannedHomePosition'] = [x[0], y[0], z[0]];
		mission['vehicleType'] = 2;
		mission['version'] = 2;
		plan['mission'] = mission;
		
		# Rally points
		rallyPoints = {};
		rallyPoints['points'] = []; 
		rallyPoints['version'] = 1;
		plan['rallyPoints'] = rallyPoints;

		# plan version
		plan['version'] = 1

		# Create JSON file
		plan_json = json.dumps(plan, indent=4, sort_keys=True)

		file = open(outFile,'w') 
		file.write (plan_json)
		file.close()
