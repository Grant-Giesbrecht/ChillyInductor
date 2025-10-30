import numpy as np
from dataclasses import dataclass

@dataclass
class datapoint:
	sigma: float #ns
	err:float 
	err_unc:float
	std_unc:float
	notes:str = ""

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

traditional_points.append(datapoint(sigma=5 , err=2.412e-2 , err_unc=2.33e-3 , std_unc=1.5e-3 ))
traditional_points.append(datapoint(sigma=10, err=3.08e-3 , err_unc=3.02e-3 , std_unc=9.01e-4 ))
traditional_points.append(datapoint(sigma=15, err=3.791e-3 , err_unc=3.793e-3 , std_unc=8.99e-4 ))
traditional_points.append(datapoint(sigma=20 , err=5.0167e-3 , err_unc=5.03e-3 , std_unc=9.53e-4 ))
traditional_points.append(datapoint(sigma=25 , err=6.63e-3 , err_unc=6.8e-3 , std_unc=1.7e-3 ))
traditional_points.append(datapoint(sigma=30 , err=7.785e-3 , err_unc=7.88e-3 , std_unc=1.46e-3 ))
traditional_points.append(datapoint(sigma=35 , err=1.0078e-2 , err_unc=1.01e-2 , std_unc=1.87e-3 ))
traditional_points.append(datapoint(sigma=40 , err=1.09e-2 , err_unc=1.11e-2 , std_unc=2.3e-3 ))
traditional_points.append(datapoint(sigma=50 , err=1.379e-2 , err_unc=1.42e-2 , std_unc=3.71e-3 ))

# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# traditional_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))

# tripler_points.append(datapoint(sigma=10 , err=2.41e-2 , err_unc=7.59e-2 , std_unc=0.125 ))
tripler_points.append(datapoint(sigma=15 , err=7.3153e-3 , err_unc=7.61e-3 , std_unc=3.12e-3 ))
tripler_points.append(datapoint(sigma=20 , err=7.63e-3 , err_unc=7.65e-3 , std_unc=2.61e-3 ))
tripler_points.append(datapoint(sigma=25 , err=8.822e-3 , err_unc=9.24e-3 , std_unc=3.26e-3 ))
tripler_points.append(datapoint(sigma=30 , err=1.0268e-2 , err_unc=1.06e-2 , std_unc=3.62e-3 ))
tripler_points.append(datapoint(sigma=35 , err=1.255e-2 , err_unc=1.29e-2 , std_unc=4.97e-3 ))
tripler_points.append(datapoint(sigma=40 , err=1.257e-2 , err_unc=1.3e-2 , std_unc=3.93e-3 ))
tripler_points.append(datapoint(sigma=50 , err=1.492e-2 , err_unc=1.56e-2 , std_unc=4.79e-3 ))
# tripler_points.append(datapoint(sigma=75 , err= , err_unc= , std_unc= ))

# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))
# tripler_points.append(datapoint(sigma= , err= , err_unc= , std_unc= ))

# datapoint(sigma=20, )

trad_sigmas = []
trad_errs = []
trad_err_uncs = []
trad_std_uncs = []
for pt in traditional_points:
	trad_sigmas.append(pt.sigma)
	trad_errs.append(pt.err)
	trad_err_uncs.append(pt.err_unc)
	trad_std_uncs.append(pt.std_unc)

doubler_sigmas = []
doubler_errs = []
doubler_err_uncs = []
doubler_std_uncs = []
for pt in doubler_points:
	doubler_sigmas.append(pt.sigma)
	doubler_errs.append(pt.err)
	doubler_err_uncs.append(pt.err_unc)
	doubler_std_uncs.append(pt.std_unc)

tri_sigmas = []
tri_errs = []
tri_err_uncs = []
tri_std_uncs = []
for pt in tripler_points:
	tri_sigmas.append(pt.sigma)
	tri_errs.append(pt.err)
	tri_err_uncs.append(pt.err_unc)
	tri_std_uncs.append(pt.std_unc)