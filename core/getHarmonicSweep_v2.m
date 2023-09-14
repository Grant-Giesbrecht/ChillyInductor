function [harm_struct, normal, Vsweep] = getHarmonicSweep_v2(rich_data, c, keep_normal)
%
%
% c: condition struct with all parameters except for harmonics
%	should include RF power and Vnrom
	
	if ~exist('keep_normal', 'var')
		keep_normal = false;
	end
	
	ld = [rich_data.dataset];

	% Add all harmoincs to c struct
	c.harms = unique([ld.harmonic]);

	% Get DC Voltages from data
	Vdcs = unique([ld.offset_V]);

	% Plot at constant SG power, sweep offset
	num_normal = 0;
	norm_V = zeros(1, numel(Vdcs));
	norm_tf = logical(zeros(1, numel(Vdcs)));
	
	% Initialize harmonic struct
	labels = ["h1", "h2", "h3", "h4", "h5"];
	for hidx = 1:numel(c.harms)
		harm_struct.(labels(hidx)) = zeros(1, numel(Vdcs));
	end

	idx = 0;
	for vdc = Vdcs
		idx = idx + 1;

		% Get harmonics at this datapoint
		c.Vdc = vdc;
		[norm, harms, err, temp] = getHarmonics_v2(rich_data, c);

		% Skip points that went normal
		if norm.pf
			num_normal = num_normal + 1;
			for hidx = 1:numel(c.harms)
				harm_struct.(labels(hidx))(idx) = NaN;
			end
		else
			% Add to lists
			for hidx = 1:numel(c.harms)
				harm_struct.(labels(hidx))(idx) = harms(hidx);
			end
		end
		
		% Add to normal output
		norm_V(idx) = norm.V;
		norm_tf(idx) = norm.pf;
	end

	% Save to output struct
	normal.pf = norm_tf;
	normal.V = norm_V;
	normal.num_normal = num_normal;
	
	Vsweep = Vdcs;
	
end