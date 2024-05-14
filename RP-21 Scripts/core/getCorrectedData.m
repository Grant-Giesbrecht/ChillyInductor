function [norm, harms, stdev] = getCorrectedData(dataset, conditions)
% GETCORRECTEDDATA 
%

%
% conditions:
%	offset_V = DC bias voltage (V)
%	SG_power_dBm = power in dBm from SG (dBm)
%	SG_freq_Hz = Frequency from generator (Hz)
%	harmonic = Harmonic to look at
%	Vnorm = threshold for going normal (V)
%	
	
	% Filter bias voltages
	filt = [dataset.offset_V];
	I1 = (filt == conditions.offset_V);

	% Filter powers
	filt = [dataset.SG_power_dBm];
	I2 = (filt == conditions.SG_power_dBm);
	
	% Filter frequencies
	filt = [dataset.SG_freq_Hz];
	I3 = (filt == conditions.SG_freq_Hz);
	
	% Filter harmonics
	filt = [dataset.harmonic];
	I4 = (filt == conditions.harmonic);
	
	% Apply mask
	mask = I1 & I2 & I3 & I4;
	points = dataset(mask);
	if numel(points) ~= numel(conditions.harms)*2 && conditions.Vdc ~= 0
		warning("Incorrect number of points found (" + num2str(numel(points)) +"). Dataset not valid.");
		displ("Occured with conditions:");
		displ("       Vdc (V): ", conditions.Vdc);
		displ("   Power (dBm): ", conditions.SG_power);
		return
	end
	
	
	
% 	% Get data for each harmonic
% 	harm_list = [points.harmonic];
% 	idx = 1;
% 	harms = zeros(1, numel(conditions.harms));
% 	stdev = zeros(1, numel(conditions.harms));
% 	for h = conditions.harms
% 		
% 		% Get relevant point
% 		ps = points(harm_list == h);
% 		
% 		mps = [];
% 		for p = ps
% 			% Get average b1/a2 value
% 			b1a2 = p.VNA_data.data(1,:); % Retreive b1/a2
% 			mag = abs(b1a2);
% 			phase_rad = angle(b1a2);
% 			
% 			% Add to list
% 			mps = [mps, polcomplex(mag, phase_rad)];
% 		end
% 		
% 		% Save result
% 		harms(idx) = mean(mps);
% 		stdev(idx) = std(mps);
% 		
% 		idx = idx + 1;
% 	end
% 	
% 	% Get readout voltage
% 	mv=[dataset(1).MFLI_voltage];
% 	voltage = mean(abs(mv));
% 	
% 	% Prepare norm output
% 	norm.pf = voltage >= conditions.Vnorm; % 'true' if is normal
% 	norm.V = voltage;	
	
end