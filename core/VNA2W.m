function pwr_w = VNA2W(harm)
	% Converts the VNA harmonic data to W
	
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	
	pwr_w = (abs(harm).*a2).^2;
	
end
