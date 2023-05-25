from colorama import Fore, Style
import numpy as np

import logging
from dataclasses import dataclass

PI = 3.1415926535

@dataclass
class LKSolution:
	Lk = None
	
	Vp = None
	betaL = None
	P0 = None
	theta = None
	Iac = None
	Zin = None
	
	L_ = None # From Lk
	Zchip = None # From L_
	
def xfmr(Z0, ZL, betaL):
	""" Calculates the input impedance looking into a transformer with characteristic 
	impedance Z0, terminated in load ZL, and electrical length betaL (radians)"""
	return Z0 * (ZL + 1j*Z0*np.tan(betaL))/(Z0 + 1j*ZL*np.tan(betaL))

class LKSystem:
	""" This class represents a solution to the nonlinear chip system, give a set of input conditions (things
	like actual chip length, input power, etc)."""
	
	def __init__(self, Pgen_dBm:float, C_:float, l_phys:float, freq:float, q:float, Ibias:list, L0:float):
		""" Initialize system with given conditions """
		
		self.Pgen = (10**(Pgen_dBm/10))/1000
		self.C_ = C_
		self.l_phys = l_phys
		self.freq = freq
		self.q = q
		self.Ibias = Ibias
		self.L0 = L0
		
		self.Zcable = 50 # Z0 of cable leading into chip
		self.Rsrc = 50 # R of generator
		self.Xsrc = 0 # X of generator
		
		# Solve for Generator voltage from power
		self.Vgen  = np.sqrt(self.Pgen*200)
		
		# Create solution object
		self.soln = LKSolution() # Current solution data
	
	
	def solve_inductance(self, Iac:float, Idc:float):
		""" Given the input current values, finds the inductance of the chip
		and updates the solution with its value."""
		
		# Solve for inductance
		# Lk_fund = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI) ) 
		
		Lk_DC = self.L0 + self.L0/self.q**2 * ( Idc**2 + Iac**2/2) 
		Lk_fund = self.L0 + self.L0/self.q**2 * ( 2*Idc*Iac ) 
		Lk_2H = self.L0 + self.L0/self.q**2 * ( Idc**2 + 2*Idc*Iac*np.sin(self.freq*2*PI) ) 
		
		
		
		
		ToDO: 
		Figure out how to handle frequency components correctly.
		What should the Lk value be when calculating all the various bits?
		
	def check_solution(self, Iac:float, Idc:float):
		""" Using the current best guess, find the solution error """
		
		
		
		self.soln.L_ = self.soln.Lk/self.l_phys
		
		self.soln.Zchip = 1/np.sqrt(self.C_ * self.soln.L_)
		
		self.soln.betaL = self.l_phys*self.freq*np.sqrt(self.C_ * self.soln.L)
		
		self.soln.Zin = xfmr(self.soln.Zchip, self.Zcable, self.soln.betaL)
		
		# Get real and imag components of Zin
		Rin = np.real(self.soln.Zin)
		Xin = np.imag(self.soln.Zin)
		
		self.soln.P0 = np.abs(self.Vgen)**2 * Rin * 0.5 / ((Rin + self.Rsrc)**2 + (Xin + self.Xsrc)**2)

