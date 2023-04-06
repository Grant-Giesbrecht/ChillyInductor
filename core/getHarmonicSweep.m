function [harm_struct, normal, Vsweep] = getHarmonicSweep(ld, c)
%
%
% c: condition struct with all parameters except for harmonics
%	should include RF power and Vnrom



	% Add all harmoincs to c struct
	c.harms = unique([ld.harmonic]);

	% Get DC Voltages from data
	Vdcs = unique([ld.offset_V]);

	% Plot at constant SG power, sweep offset
	num_normal = 0;
	norm_V = zeros(1, numel(Vdcs));
	norm_tf = logical(zeros(1, numel(Vdcs)));
	h1 = zeros(1, numel(Vdcs));
	h2 = zeros(1, numel(Vdcs));
	h3 = zeros(1, numel(Vdcs));
	h4 = zeros(1, numel(Vdcs));
	h5 = zeros(1, numel(Vdcs));

	idx = 0;
	for vdc = Vdcs
		idx = idx + 1;

		% Get harmonics at this datapoint
		c.Vdc = vdc;
		[norm, harms, err] = getHarmonics(ld, c);

		% Skip points that went normal
		if norm.pf
			num_normal = num_normal + 1;
			harms = [NaN, NaN, NaN, NaN, NaN];
		end
		
		% Add to normal output
		norm_V(idx) = norm.V;
		norm_tf(idx) = norm.pf;

		% Add to lists
		harm_struct.h1(idx) = harms(1);
		harm_struct.h2(idx) = harms(2);
		harm_struct.h3(idx) = harms(3);
		harm_struct.h4(idx) = harms(4);
		harm_struct.h5(idx) = harms(5);


	end

	% Save to output struct
	normal.pf = norm_tf;
	normal.V = norm_V;
	normal.num_normal = num_normal;
	
	Vsweep = Vdcs;
	
end