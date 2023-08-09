function [I, V] = ZPwr2IV(Zline, P_W)
	% Accepts an impedance (can be an impedance at a certain point along a
	% line) and returns the V and I coefficients (including phase) that
	% delivers P_W watts.
	
	% Find angle
	Z_arg = angle(Zline);
	
	% Return V and I
	I = sqrt(2.*P_W./cos(Z_arg)./Zline);
	V = sqrt(2.*P_W./cos(Z_arg).*Zline);
	
	
end