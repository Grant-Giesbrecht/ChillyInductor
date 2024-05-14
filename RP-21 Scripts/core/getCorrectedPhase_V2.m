function [corrected_phase, data_vbias] = getCorrectedPhase_V2(rich_data, P_RF, f0, plot_phase_correction, normalize, take_avg)
%      phase, vbias, dp_idx, freq, take_avg, normalize, plot_phasecorrection)

% Given a list of phases, bias voltages, and collection indecies, it
% returns the unwrapped and drift-corrected phase data. take_avg allows
% sweep2mean to be automatically applied, and plot_phase_correction plots
% intermediate graphs, and normalize sets the zero bias phase to zero
% degrees.
%
% This assumes data has ALREADY BEEN FILTERED TO 1 FREQUENCY
%
	
	% Default arguments
	if ~exist('take_avg', 'var')
		take_avg = false;
	end
	
	if ~exist('normalize', 'var')
		normalize = true;
	end
	if ~exist('plot_phase_correction', 'var')
		plot_phase_correction = true;
	end

	%% Prepare data from V2 format to basic lists
	
	% Filter frequency
	freqs = [rich_data.dataset.SG_freq_Hz];
	I_freq = (freqs == f0);
	
	% Grab first harmonic only
	harms = [rich_data.dataset.harmonic];
	I_harm = (harms == 1);
	
	% Filter power
	powers = [rich_data.dataset.SG_power_dBm];
	I_pwr = (powers == P_RF);
	
	% Generate mask
	mask = I_freq & I_harm & I_pwr;
	sel_data = [rich_data.dataset(mask)];
	
	% Get all phases
	phase = zeros(1, numel(sel_data));
	idx = 1;
	for dp = sel_data
		
		phase(idx) = mean(angle(dp.VNA_data.data(1,:)));
		
		idx = idx + 1;
	end
	phase = phase .* 180 ./ pi;
	
	% Get all bias voltages
	vbias = [sel_data.offset_V];
	dp_idx = [sel_data.collection_index];
	
	freq = f0;
	
	%% Break Data up by Frequency, Control and Main
	%
	% Loops over all datapoints and creates structs:
	%	ctrl_phase: Phase at each point (Control set, bias = 0 V)
	%	ctrl_dpidx: Collection index (Control set, bias = 0 V)
	%
	%	data_phase: Phase of point
	%	data_dpidx: Collection index
	%	data_vbias: Bias voltage
	%
	% Each struct has fields for each frequency. 10 GHz for example would be:
	%	data_phase.f10
	
	% Get control data
	ctrl_idx_s = (vbias == 0);
	ctrl_phase = phase(ctrl_idx_s);
	ctrl_dpidx = dp_idx(ctrl_idx_s);
	
	figure(21);
	subplot(3, 1, 1);
	NP= numel([sel_data.offset_V]);
	idxes = 1:NP;
	hold off;
	plot(idxes, [sel_data.offset_V], 'Marker', '.', 'LineStyle', ':');
	xlabel("Collection Index");
	ylabel("Bias Voltage (V)");
	grid on;
	xlim([0, NP]);

	subplot(4, 1, 2);
	hold off;
	plot(idxes, [sel_data.SG_power_dBm], 'Marker', '.', 'LineStyle', ':');
	xlabel("Collection Index");
	ylabel("RF Power (dBm)");
	grid on;
	xlim([0, NP]);

	subplot(4, 1, 3);
	hold off;
	plot(idxes, [sel_data.SG_freq_Hz]./1e9, 'Marker', '.', 'LineStyle', ':');
	xlabel("Collection Index");
	ylabel("Fundamental Frequency (GHz)");
	grid on;
	xlim([0, NP]);
	
	% Find datapoint for 0 bias (actual measurement, not phase tracking
	% point). If bias goes negative, get the zero point when the sign
	% changes.
	if any(rich_data.configuration.bias_V < 0)

		% Find zero point - these are the points where the actual data should be at
		% 0 bias. These points occur when the Vbias sign flips. There should be 2
		% points.
		
		vb_d = [];
		idx_d = [];
		
		zp_idx = [];
		all_zeros = find(ctrl_idx_s);
		for idx = all_zeros

			% Verify in bounds
			if ~(idx-2 > 0 && idx+2 <= numel(phase))
				continue;
			end
			
			idx_d = [idx_d, idx];
			vb_d = [vb_d, vbias(idx-2)/vbias(idx+2)];
			
			% Will be negative if sign flipped
			if vbias(idx-2)/vbias(idx+2) < 0
				zp_idx = [zp_idx, idx];
			end
		end
		
		subplot(4, 1, 4);
		hold off;
		plot(idx_d, vb_d, 'Marker', '.', 'LineStyle', ':');
		xlabel("Collection Index");
		ylabel("V_b[-1]/V_b[+1] (1)");
		grid on;
		xlim([0, NP]);
		
		% Check for correct number of zeropoints found
		if numel(zp_idx) ~= 2
			warning("Failed to find zero points");
			return;
		end
		
	% If the bias never goes negative, get the first zero point.
	else
		
		all_zeros = find(ctrl_idx_s);
		zp_idx = [all_zeros(1), all_zeros(1)];
		
	end

	% Get primary data
	data_idx_s = find(~ctrl_idx_s); % Invert control selection
	data_idx_s = [data_idx_s, zp_idx]; % Add zero points
	data_idx_s = sort(data_idx_s); % Re-sort

	data_phase = phase(data_idx_s);
	data_dpidx = dp_idx(data_idx_s);
	data_vbias = vbias(data_idx_s);


% 	%% For each frequency point, produce estimates for N and el/V0
% 	%
% 	% This estimate could by done by guessing numbers of wraps, then picking
% 	% the one that has the closest remainder to the measured phase. However,
% 	% isntead I'm estimating wavelength and guessing how many 'fit' in the
% 	% estimated length.
% 	%
% 	%
% 	% This will produce a list:
% 	%	N_guess: Guessed number of wraps for each frequency. Indexing matches
% 	%	'frequencies'.
% 
% 	% From calc_Len_auto.m, I've estimated:
% 	%	VF: 0.7645
% 	%	Length: 1.363 m
% 	% From there we know that l/v0 should be, so we'll pick the N for each
% 	% frequency that gives the closest l/v0 match.
% 	VF_est = 0.76;
% 	l_est = 1.4;
% 	% VF_est = .0654;
% 
% 	N_guess = zeros(1, numel(frequencies));
% 
% 	% Scan over all frequencies and estimate T and el/V0
% 	idx = 0;
% 	for f = frequencies
% 		idx = idx + 1;
% 
% 		T = 1/f; % Period (s)
% 		lambda = 3e8/f/VF_est; % Wavelength (m)
% 		n = l_est/lambda; % No. wavelength
% 		N = floor(n); % No. wraps
% 
% 		% Save guess
% 		N_guess(idx) = N;
% 	end
% 
% 	figure(20);
% 	plot(frequencies./1e9, N_guess, 'LineStyle', '--', 'Marker', '*')
% 	xlabel("Frequency (GHz)");
% 	ylabel("Number of Wraps");
% 	grid on;

	%% Apply Correction
	%
	% For each analyzed frequency, find:
	%	1.) Phase correction from "control" data
	%	2.) When wraps occur due to bias, giving *absolute phase*.
	%	--> Plot phase with drift correction and absolute phase.
	%	3.) Get dT from phase and period.
	%	4.) Apply dT to find 'x' scaling coefficient.

	

	%=========== (1.) Get Phase Correction from Control Data ==============

	% Normalize data
	mean_phase = mean(ctrl_phase); % Find mean - apply correction to make all phases equal mean
	cf_offset = mean_phase - ctrl_phase; % Add this to each point to 'correct' phase

	% Plot Data
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 1);
		hold off;
		scatter(ctrl_dpidx, cf_offset);
		grid on;
		xlabel("Collection Index");
		ylabel("Phase Correction (deg)");
		title(strcat("Phase Correction, Freq = ", num2str(freq/1e9), " GHz"));
	end

	%======= (2.) Find phase wraps from bias, get absolute phase ==========

	% Unwrap phase
	phase_uw = unwrap(data_phase./180.*pi).*180./pi;
	
	% Plot unwrapped Phase
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 2);
		hold off;
		scatter(data_vbias, data_phase, 'Marker', 'o');
		hold on;
		scatter(data_vbias, phase_uw, 'Marker', '*');
		grid on;
		xlabel("Collection Index");
		ylabel("Phase Correction (deg)");
		legend("Raw", "Unwrapped");
		title(strcat("Unwrapped Phase, Freq = ", num2str(freq/1e9), " GHz"));
	end

	% Loop over each point and calculate correction
	idx = 0;
	correction = zeros(1, numel(data_dpidx));
	for cidx = data_dpidx % Loop over each data point's collection index
		idx = idx + 1;
		if cidx-1 < min(ctrl_dpidx)
			norm_idx1 = cidx + 1;
			I1 = (ctrl_dpidx == norm_idx1);
			correction(idx) = cf_offset(I1);
		elseif cidx+1 > max(ctrl_dpidx)
			norm_idx0 = cidx - 1;
			I0 = (ctrl_dpidx == norm_idx0);
			correction(idx) = cf_offset(I0);
		elseif any(ctrl_dpidx == cidx)
			I0 = (ctrl_dpidx == cidx);
			correction(idx) = cf_offset(I0);
		else
			norm_idx0 = cidx - 1;
			norm_idx1 = cidx + 1;

			% Find points to average
			try
				I0 = (ctrl_dpidx == norm_idx0);
				I1 = (ctrl_dpidx == norm_idx1);
				correction(idx) = (cf_offset(I0) + cf_offset(I1))/2;
			catch
				correction(idx) = (cf_offset(I0) + cf_offset(I1))/2;
			end
		end
	end

	% Apply normalization
	corrected_phase = phase_uw + correction;

% 	% Add N extra wavelength
% 	In = (freq./1e9 == freq); % Get index
% 	absolute_phase = (N_guess(In)+1)*360 - corrected_phase;

	% Plot corrected Phase
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 3);
		hold off;
		scatter(data_vbias, data_phase, 'Marker', 'o');
		hold on;
		scatter(data_vbias, phase_uw, 'Marker', '*');
		scatter(data_vbias, corrected_phase, 'Marker', '*', 'MarkerFaceColor', [0, .7, 0], 'MarkerEdgeColor', [0, .7, 0]);
		grid on;
		xlabel("Bias Voltage (V)");
		ylabel("Phase Correction (deg)");
		legend("Raw", "Unwrapped", "Unwrapped and drift corrected");
		title(strcat("Unwrapped Phase, Freq = ", num2str(freq/1e9), " GHz"));
	end

	% Average each phase for each unique bias voltage
	if take_avg
		
		% Apply averaging
		[udvb, mean_phase] = sweep2mean(data_vbias, corrected_phase);
		
		% Save to output variables
		data_vbias = udvb;
		corrected_phase = mean_phase;
	end
	
	% Normalize if requested
	if normalize
		norm_idx = find(data_vbias == 0, 1);
		corrected_phase = corrected_phase - corrected_phase(norm_idx);
	end

end












