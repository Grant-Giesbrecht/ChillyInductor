% This script is built-off plot_drift_dataset.m, but contains frequency
% sweeps and finer resolution sweeps. 
%
%

load(fullfile("","","FOS1_Data.mat"));

plot_freqs = [1, 5, 50]; % Frequencies to plot [GHz]


plot_each_set = true; % Fig. 10

plot_phase_correction = true; % Fig. 30

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

dp_idx = datapoint_index;
phase = Pb1a2;
vbias = Vbias;

% Get unique frequencies
frequencies = unique(freq);

% Process each frequency
for f = frequencies
	
	% Find indecies for this frequency
	fIdx = (freq == f);
	
	struct_key = string(strcat('f', num2str(f/1e9)));
	
	dp_idx_s = dp_idx(fIdx);
	vbias_s = vbias(fIdx);
	phase_s = phase(fIdx);
	
	% Plot all data
	if plot_each_set
		figure(10);
		hold off;
		scatter3(dp_idx_s, vbias_s, phase_s)
		xlabel("Collection Index");
		ylabel("Bias Voltage (V)");
		zlabel("Phase (deg)");
		title(strcat("Frequency = ", num2str(f./1e9), " GHz") );
	end
	
	% Get control data
	ctrl_idx_s = (vbias_s == 0);
	ctrl_phase.(struct_key) = phase_s(ctrl_idx_s);
	ctrl_dpidx.(struct_key) = dp_idx_s(ctrl_idx_s);

	% Find zero point - these are the points where the actual data should be at
	% 0 bias. These points occur when the Vbias sign flips. There should be 2
	% points.
	zp_idx = [];
	all_zeros = find(ctrl_idx_s);
	for idx = all_zeros

		% Verify in bounds
		if ~(idx-1 > 0 && idx+1 <= numel(phase))
			continue;
		end

		% Will be negative if sign flipped
		if vbias_s(idx-1)/vbias_s(idx+1) < 0
			zp_idx = [zp_idx, idx];
		end
	end

	% Check for correct number of zeropoints found
	if numel(zp_idx) ~= 2
		warning("Failed to find zero points");
		return;
	end

	% Get primary data
	data_idx_s = find(~ctrl_idx_s); % Invert control selection
	data_idx_s = [data_idx_s, zp_idx]; % Add zero points
	data_idx_s = sort(data_idx_s); % Re-sort

	data_phase.(struct_key) = phase_s(data_idx_s);
	data_dpidx.(struct_key) = dp_idx_s(data_idx_s);
	data_vbias.(struct_key) = vbias_s(data_idx_s);
	
end

%% For each frequency point, produce estimates for N and el/V0
%
% This estimate could by done by guessing numbers of wraps, then picking
% the one that has the closest remainder to the measured phase. However,
% isntead I'm estimating wavelength and guessing how many 'fit' in the
% estimated length.
%
%
% This will produce a list:
%	N_guess: Guessed number of wraps for each frequency. Indexing matches
%	'frequencies'.

% From calc_Len_auto.m, I've estimated:
%	VF: 0.7645
%	Length: 1.363 m
% From there we know that l/v0 should be, so we'll pick the N for each
% frequency that gives the closest l/v0 match.
VF_est = 0.76;
l_est = 1.4;
% VF_est = .0654;

N_guess = zeros(1, numel(frequencies));

% Scan over all frequencies and estimate T and el/V0
idx = 0;
for f = frequencies
	idx = idx + 1;
		
	T = 1/f; % Period (s)
	lambda = 3e8/f/VF_est; % Wavelength (m)
	n = l_est/lambda; % No. wavelength
	N = floor(n); % No. wraps
	
	% Save guess
	N_guess(idx) = N;
end

figure(20);
plot(frequencies./1e9, N_guess, 'LineStyle', '--', 'Marker', '*')
xlabel("Frequency (GHz)");
ylabel("Number of Wraps");
grid on;

%% Get Nonlinearity Metric
%
% For each analyzed frequency, find:
%	1.) Phase correction from "control" data
%	2.) When wraps occur due to bias, giving *absolute phase*.
%	--> Plot phase with drift correction and absolute phase.
%	3.) Get dT from phase and period.
%	4.) Apply dT to find 'x' scaling coefficient.

for fp = plot_freqs
	
	%=========== (1.) Get Phase Correction from Control Data ==============
	
	% Define struct field name
	struct_key = string(strcat('f', num2str(fp)));
	
	% Get data from structs
	cf_phase = ctrl_phase.(struct_key);
	cf_dpidx = ctrl_dpidx.(struct_key);
	
	% Normalize data
	mean_phase = mean(cf_phase); % Find mean - apply correction to make all phases equal mean
	normalization_index = ceil(numel(cf_phase)/2); % Find center index
	cf_offset = mean_phase - cf_phase; % Add this to each point to 'correct' phase

	% Plot Data
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 1);
		hold off;
		scatter(cf_dpidx, cf_offset);
		grid on;
		xlabel("Collection Index");
		ylabel("Phase Correction (deg)");
		title(strcat("Phase Correction, Freq = ", num2str(fp), " GHz"));
	end
	
	%======= (2.) Find phase wraps from bias, get absolute phase ==========
	
	% Get data from structs
	df_phase = data_phase.(struct_key);
	df_vbias = data_vbias.(struct_key);
	df_dpidx = data_dpidx.(struct_key);
	
	% Unwrap phase
	phase_uw = unwrap(df_phase./180.*pi).*180./pi;
	
	% Plot unwrapped Phase
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 2);
		hold off;
		scatter(df_vbias, df_phase, 'Marker', 'o');
		hold on;
		scatter(df_vbias, phase_uw, 'Marker', '*');
		grid on;
		xlabel("Collection Index");
		ylabel("Phase Correction (deg)");
		legend("Raw", "Unwrapped");
		title(strcat("Unwrapped Phase, Freq = ", num2str(fp), " GHz"));
	end
	
	% Loop over each point and calculate correction
	idx = 0;
	correction = zeros(1, numel(df_dpidx));
	for cidx = df_dpidx % Loop over each data point's collection index
		idx = idx + 1;
		if cidx-1 < min(cf_dpidx)
			norm_idx1 = cidx + 1;
			I1 = (cf_dpidx == norm_idx1);
			correction(idx) = cf_offset(I1);
		elseif cidx+1 > max(cf_dpidx)
			norm_idx0 = cidx - 1;
			I0 = (cf_dpidx == norm_idx0);
			correction(idx) = cf_offset(I0);
		elseif any(cf_dpidx == cidx)
			I0 = (cf_dpidx == cidx);
			correction(idx) = cf_offset(I0);
		else
			norm_idx0 = cidx - 1;
			norm_idx1 = cidx + 1;

			% Find points to average
			try
				I0 = (cf_dpidx == norm_idx0);
				I1 = (cf_dpidx == norm_idx1);
				correction(idx) = (cf_offset(I0) + cf_offset(I1))/2;
			catch
				correction(idx) = (cf_offset(I0) + cf_offset(I1))/2;
			end
		end
	end
	
	% Apply normalization
	corrected_phase = phase_uw + correction;
	
	% Add N extra wavelength
	In = (frequencies./1e9 == fp); % Get index
	absolute_phase = (N_guess(In)+1)*360 - corrected_phase;
	
	% Plot corrected Phase
	if plot_phase_correction
		figure(30);
		subplot(1, 3, 3);
		hold off;
		scatter(df_vbias, df_phase, 'Marker', 'o');
		hold on;
		scatter(df_vbias, phase_uw, 'Marker', '*');
		scatter(df_vbias, corrected_phase, 'Marker', '*', 'MarkerFaceColor', [0, .7, 0], 'MarkerEdgeColor', [0, .7, 0]);
		grid on;
		xlabel("Bias Voltage (V)");
		ylabel("Phase Correction (deg)");
		legend("Raw", "Unwrapped", "Unwrapped and drift corrected");
		title(strcat("Unwrapped Phase, Freq = ", num2str(fp), " GHz"));
	end
	
	figure(2);
	hold off;
	scatter(df_vbias, absolute_phase, 'Marker', '*', 'MarkerFaceColor', [0, .7, 0], 'MarkerEdgeColor', [0, .7, 0]);
	xlabel("Bias Voltage (V)");
	ylabel("Absolute Phase (deg)");
	title(strcat("Corrected Absolute Phase, Freq = ", num2str(fp), " GHz"));
	grid on;
	
	
	%================ (3.) Get dT from Phase and Period ===================
	
	T = 1/(fp*1e9); % Get period (s)
	dT = absolute_phase.*T./360; % Get time elapsed (s)
		
	%========== (4.) Apply dT to find 'x' scaling coefficient =============
	
	% At 1st 0V point, estimate l/V0
	I_0V = find(df_vbias == 0); % Find index
	I_0V = I_0V(1);
	lv0 = dT(I_0V); % Pull time from that index
	displ("t0 = ", lv0.*1e9, " ns");
	
	% Calculate x everywhere
	x = ( dT./lv0 ).^2;
	
	figure(3);
	hold off;
	plot(df_vbias, x, 'LineStyle', ':', 'Marker', 'o');
	xlabel("Bias Voltage (V)");
	ylabel("Inductor Scaling");
	title(strcat("Inductor Nonlinearity, Freq = ", num2str(fp), " GHz"));
	grid on;
	
	input("Enter to continue");
	
end







