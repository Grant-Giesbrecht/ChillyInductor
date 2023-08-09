function pwr_dBm = VNA2dBm(harm)
	% Converts the VNA harmonic data to dBm
	
	pw = VNA2W(harm);
	pwr_dBm = cvrt(pw, 'W', 'dBm');
	
end
