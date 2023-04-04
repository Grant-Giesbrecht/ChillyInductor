function [norm, harms, stdev] = getHarmonics(dataset, conditions)
% GETHARMONICS 
%
% Returns the power and phase of harmonics at a given set of input
% conditions (most often, the bias voltage is the swept input condition).
% If multiple datapoints are available for a set of conditions (ie. your
% data were repeated in order to try to cancel out drift), the points are
% averaged (ie. no drift cancelation is done other than averaging). 
%
% 'norm' contains both a pass/fail value (pf) which is true if the chip
% goes normal under these bias conditions, and and the actual voltage
% value (V). 
%
% 'harms' contains one complex value for each requested harmonic.
%
% 'stdev' contains the standard deviation of the averaged values that goes
% into each harmonic complex value. (ie. 1 stdev value for each harmonic).
%
% conditions:
%	SG_power = power in dBm from SG (dBm)
%	Vdc = DC bias voltage (V)
%	harms = Harmonics to look at
%	Vnorm = threshold for going normal (V)	
	
	
	% Filter powers
	pwr = [dataset.SG_power_dBm];
	I1 = (pwr == conditions.SG_power);
	
	% Filter bias voltages
	bias = [dataset.offset_V];
	I2 = (bias == conditions.Vdc);
	
	% Apply mask
	mask = I1 & I2;
	points = dataset(mask);
	if numel(points) ~= numel(conditions.harms)*2 && conditions.Vdc ~= 0
		warning("Incorrect number of points found (" + num2str(numel(points)) +"). Dataset not valid.");
		displ("Occured with conditions:");
		displ("       Vdc (V): ", conditions.Vdc);
		displ("   Power (dBm): ", conditions.SG_power);
		return
	end
	
	% Get data for each harmonic
	harm_list = [points.harmonic];
	idx = 1;
	harms = zeros(1, numel(conditions.harms));
	stdev = zeros(1, numel(conditions.harms));
	for h = conditions.harms
		
		% Get relevant point
		ps = points(harm_list == h);
		
		mps = [];
		for p = ps
			% Get average b1/a2 value
			b1a2 = p.VNA_data.data(1,:); % Retreive b1/a2
			mag = abs(b1a2);
			phase_rad = angle(b1a2);
			
			% Add to list
			mps = [mps, polcomplex(mag, phase_rad)];
		end
		
		% Save result
		harms(idx) = mean(mps);
		stdev(idx) = std(mps);
		
		idx = idx + 1;
	end
	
	% Get readout voltage
	mv=[dataset(1).MFLI_voltage];
	voltage = mean(abs(mv));
	
	% Prepare norm output
	norm.pf = voltage >= conditions.Vnorm; % 'true' if is normal
	norm.V = voltage;	
	
end