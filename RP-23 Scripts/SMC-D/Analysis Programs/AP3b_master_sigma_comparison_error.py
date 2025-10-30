import numpy as np
from dataclasses import dataclass

@dataclass
class datapoint:
	sigma: float #ns
	err:float 
	err_unc:float
	std_unc:float
	notes:str

# List of datapoints
traditional_points = []
doubler_points = []
tripler_points = []

#=========== Add doubler points ==============
doubler_points.append(datapoint(sigma=5, err=5.36e-3, err_unc=8.64e-3, std_unc=1.18e-2))
doubler_points.append(datapoint(sigma=10, err=3.614e-3, err_unc=3.47e-3, std_unc=1.56e-3))
doubler_points.append(datapoint(sigma=15, err=4.387e-3, err_unc=4.38e-3, std_unc=1.23e-3))
doubler_points.append(datapoint(sigma=20, err=5.46e-3, err_unc=5.5e-3, std_unc=1.41e-3))
doubler_points.append(datapoint(sigma=25, err=6.987e-3, err_unc=7.03e-3, std_unc=1.53e-3 ))
doubler_points.append(datapoint(sigma=30 , err=8.391e-3 , err_unc=8.66e-3 , std_unc=2.26e-3 ))
doubler_points.append(datapoint(sigma=35, err=9.598e-3 , err_unc=9.82e-3 , std_unc=2.227e-3 ))
doubler_points.append(datapoint(sigma=40 , err=1.09e-2 , err_unc=1.13e-2 , std_unc=2.41e-3 ))
doubler_points.append(datapoint(sigma=50 , err=1.405e-2 , err_unc=1.44e-2 , std_unc=3.46e-3 ))
# doubler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# doubler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# doubler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# doubler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))

# datapoint(sigma=20, )

