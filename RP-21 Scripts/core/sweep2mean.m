function [unique_bias, mean_phase] = sweep2mean(data_vbias, phase)
% SWEEP2MEAN Takes a sweep with multiple points for each bias voltage, and
% returns a mean for each unique voltage.
%
%	SWEEP2MEAN(data_vbias, phase) 'data_vbias' contains the bias
%	voltages for each point (N values). 'phase' contains the phase at each
%	point (N values). Returns a list of the unique bias voltages and mean 
%	of the corresponding phase points for each bias voltage.
%

	% Find unique bias voltages
	unique_bias = unique(data_vbias);

	% Create averaged data array
	mean_phase = zeros(1, numel(unique_bias));

	% Loop over each unique bias voltage
	idx = 0;
	for vb = unique_bias
		idx = idx + 1;

		% Filter points
		Ivb = (data_vbias == vb);
		mean_phase(idx) = mean(phase(Ivb));
	end
end