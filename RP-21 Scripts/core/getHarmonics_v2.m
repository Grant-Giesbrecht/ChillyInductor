function [norm, harms, stdev, temp] = getHarmonics_v2(rich_data, conditions)
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
%	f0 = Fundamental frequency (Hz) (-1 for select all)
%	Vdc = DC bias voltage (V) 
%	harms = Harmonics to look at
%	Vnorm = threshold for going normal (V)
%	convert_to_W  = (logical) convert from VNA b1a2 readings to watts
	
	dataset = [rich_data.dataset];
	
	% Filter powers
	pwr = [dataset.SG_power_dBm];
	I1 = (pwr == conditions.SG_power);
	
	% Filter bias voltages
	bias = [dataset.offset_V];
	I2 = (bias == conditions.Vdc);
	
	% Filter fundamental frequencies
	freqs = [dataset.SG_freq_Hz];
	if conditions.f0 ~= -1
		I3 = (freqs == conditions.f0);
	else
		I3 = ones(1, numel(freqs));
	end
	
	% Apply mask
	mask = I1 & I2 & I3;
	points = dataset(mask);
	if rich_data.configuration.duplicate_reverse
		if numel(points) ~= numel(conditions.harms)*2 && conditions.Vdc ~= 0

			warning("Incorrect number of points found (" + num2str(numel(points)) +"). Dataset not valid.");
			displ("Occured with conditions:");
			displ("       Vdc (V): ", conditions.Vdc);
			displ("   Power (dBm): ", conditions.SG_power);
			displ("   Freq (GHz): ", conditions.f0/1e9);
			norm.pf = [];
			norm.V = [];
			harms = [];
			stdev=[];
			temp=[];
			return
		end
	else
		if numel(points) ~= numel(conditions.harms) && conditions.Vdc ~= 0

			warning("Incorrect number of points found (" + num2str(numel(points)) +"). Dataset not valid.");
			displ("Occured with conditions:");
			displ("       Vdc (V): ", conditions.Vdc);
			displ("   Power (dBm): ", conditions.SG_power);
			displ("   Freq (GHz): ", conditions.f0/1e9);
			norm.pf = [];
			norm.V = [];
			harms = [];
			stdev=[];
			temp=[];
			return
		end
	end
	
	% Get data for each harmonic
	harm_list = [points.harmonic];
	idx = 1;
	harms = zeros(1, numel(conditions.harms));
	clear temp;
	stdev = zeros(1, numel(conditions.harms));
	for h = conditions.harms
		
		% Get relevant point
		ps = points(harm_list == h);
		
		temps = [];
		mps = [];
		for p = ps
			% Get average b1/a2 value
			b1a2 = p.VNA_data.data(1,:); % Retreive b1/a2
			mag = abs(b1a2);
			phase_rad = angle(b1a2);
			
			% Add to list
			mps = [mps, polcomplex(mag, phase_rad)];
			temps = [temps, p.temp_K];
		end
		
		% Save result
		if conditions.convert_to_W
			mps = VNA2W(mps);
		end
		harms(idx) = mean(mps);
		stdev(idx) = std(mps);
		if idx == 1
			temp(numel(conditions.harms)) = struct('temp_K', mean(temps), 'Temp_raw_K', temps);
		end
		temp(idx) = struct('temp_K', mean(temps), 'Temp_raw_K', temps);
		
		idx = idx + 1;
	end
	
	% Get readout voltage
	mv=[points.MFLI_voltage];
% 	mv=[points.MFLI_voltage_deg];
% 	mv=abs([points.MFLI_voltage_V] + 1i.*[points.MFLI_voltage_deg]);
	voltage = mean(abs(mv));
	
	% Prepare norm output
	norm.pf = voltage >= conditions.Vnorm; % 'true' if is normal
	norm.V = voltage;	
	
end