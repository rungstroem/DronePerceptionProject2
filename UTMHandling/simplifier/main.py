
## Import class
from simplifyClass import simplifyClass

# Initialize class
simplify = simplifyClass;
#simplify.maxWaypoints(10, "UTM_positions_corrected.csv");
simplify.douglasPeucker("UTM_positions_corrected.csv", 0.5);
