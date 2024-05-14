%% Load datafile

if ~exist('ds', 'var')
	load(fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023',...
		'Data', 'group4_extflash', 'December data', 'beta_10GHz_V1_07Dec2023.mat'));
	
	
end
keep_idxs = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50];
analysis_freqs = ds.configuration.frequency;
P_RF = 0;

% Save manual unwrap commands
CMD = struct();
CMD.f0 = 9.53e9;
CMD.bias_low_V = -0.4;
CMD.bias_high_V = 0.2;
% This phase number (as opposed to 180) was chosen as it makes the ends of q
% flat, and thus is probably right.
CMD.phase_correction = 153;

% Apply bias limits
LIMIT_BIAS = true;
bias_min_V = 0.5;
bias_max_V = 1.5;


%% View Dataset Sweep

ZOOM_ONE_FREQ = false;

f1=figure(1);
f1.Position = [1 573 410 444];
subplot(3, 1, 1);
NP= numel([ds.dataset.offset_V]);
idxes = 1:NP;
hold off;
plot(idxes, [ds.dataset.offset_V], 'Marker', '.', 'LineStyle', ':');
xlabel("Collection Index");
ylabel("Bias Voltage (V)");
grid on;
xlim([0, NP]);

subplot(3, 1, 2);
hold off;
plot(idxes, [ds.dataset.SG_power_dBm], 'Marker', '.', 'LineStyle', ':');
xlabel("Collection Index");
ylabel("RF Power (dBm)");
grid on;
xlim([0, NP]);

subplot(3, 1, 3);
hold off;
plot(idxes, [ds.dataset.SG_freq_Hz]./1e9, 'Marker', '.', 'LineStyle', ':');
xlabel("Collection Index");
ylabel("Fundamental Frequency (GHz)");
grid on;
xlim([0, NP]);

if ZOOM_ONE_FREQ
	idx_f2 = find([ds.dataset.SG_freq_Hz] == ds.configuration.frequency(2), 1, 'first');
	for IDX_ = 1:3
		subplot(3, 1, IDX_);
		xlim([0, idx_f2-1]);
	end
end

%% Scan over all frequncies and analyze

MASTER_DATA = struct();

for freq_idx = keep_idxs
	
	struct_key = "Fidx"+num2str(freq_idx);
	
	%% Calculate raw phases and powers

	%-------------------------- Select what to plot ---------------------------
	P_RF_rawplot = P_RF;
	f0_rawplot = analysis_freqs(freq_idx);
	displ(f0_rawplot./1e9, "e9");

	%-------------------------------------------------------------------------

	% Filter frequency
	freqs = [ds.dataset.SG_freq_Hz];
	I_freq = (freqs == f0_rawplot);

	% Grab first harmonic only
	harms = [ds.dataset.harmonic];
	I_harm = (harms == 1);

	% Filter power
	powers = [ds.dataset.SG_power_dBm];
	I_pwr = (powers == P_RF_rawplot);

	% Generate mask
	mask = I_freq & I_harm & I_pwr;
	sel_data = [ds.dataset(mask)];

	% Get all phases
	raw_phases = zeros(1, numel(sel_data));
	raw_biases = zeros(1, numel(sel_data));
	raw_mags = zeros(1, numel(sel_data));
	raw_ci= zeros(1, numel(sel_data));
	idx = 1;
	for dp = sel_data

		raw_phases(idx) = mean(angle(dp.VNA_data.data(1,:)));
		raw_mags(idx) = mean(abs(dp.VNA_data.data(1,:)));
		raw_biases(idx) = dp.offset_V;
	% 	raw_powers(idx) = dp.SG_power_dBm;
		raw_ci(idx) = dp.collection_index;

		idx = idx + 1;
	end
	raw_phases = raw_phases .* 180 ./ pi;

	raw_mags_mW = cvrt(raw_mags, 'dBm', 'mW').*1e3;

	%% Pick which points have valid phases
	%
	% This is based on the magnitude of the signal received. The threshold for
	% valid or not valid is 50% between the lowest and highest powers measured.
	% This tends to be a pretty clear distinction.

	pwr_threshold = (max(raw_mags_mW) + min(raw_mags_mW))/2;

	I_valid = (raw_mags_mW > pwr_threshold);

	f2=figure(2);
	f2.Position = [412 716 570 301];
	hold off;
	plot3(raw_biases, raw_ci, raw_mags_mW, 'Marker', '.', 'LineStyle', ':', 'Color', [0, 0.7, 0]);
	hold on;
	scatter3(raw_biases(I_valid), raw_ci(I_valid), raw_mags_mW(I_valid), 'Marker', 'o',...
		'MarkerEdgeColor', [0.7, 0, 0]);

	% Draw plane for threshold power
	Xs = [min(raw_biases), min(raw_biases), max(raw_biases), max(raw_biases)];
	Ys = [min(raw_ci), max(raw_ci), max(raw_ci), min(raw_ci)];
	Zs = [pwr_threshold, pwr_threshold, pwr_threshold, pwr_threshold];
	patch(Xs, Ys, Zs, [0, 0, 0.8], 'FaceAlpha', 0.1);

	grid on;
	xlabel("Bias Voltage (V)");
	ylabel("Collection Index");
	zlabel("Received Power (mW)");
	title("Selection of Valid Phase Points");
	legend("All Data Points", "Selected Data Points", 'Position', [0.2026, 0.4250, 0.2211, 0.088]);

	%% Calculate averaged (ie. phase corrected) phases

	raw_biases_sel = raw_biases(I_valid);
	raw_phases_sel = raw_phases(I_valid);

	% Take averages over biases
	avgd_bias = unique(raw_biases_sel);
	avg_phase= zeros(1, numel(avgd_bias));
	idx = 0;
	for bv = avgd_bias
		idx = idx +1;

		mask = (raw_biases_sel == bv);
		avg_phase(idx) = mean(raw_phases_sel(mask));

	end

	f3=figure(3);
	f3.Position = [1 73 560 420];
	hold off;
	plot3(raw_biases_sel, raw_ci(I_valid), raw_phases(I_valid), 'Marker', '.',...
		'LineStyle', ':', 'Color', [0.9, 0.8, 0.3], 'MarkerSize', 13);
	hold on;
	plot3(avgd_bias, ones(1, numel(avgd_bias)), avg_phase, 'Marker', '.', 'LineStyle',...
		':', 'Color', [0, 0, 0.8], 'MarkerSize', 13);
	view([0,-1,0]);
	legend("Raw Data", "Averaged Data (Phase drift correction)", 'Location', 'Best');

	grid on;
	xlabel("Bias Voltage (V)");
	ylabel("Collection Index");
	zlabel("Phase (^\circ)");
	title("Selection of Valid Phase Points");
	force0z;

	%% Add section to unwrap phase?

	% Auto unwrap
	avgd_phase = unwrap(avg_phase./180.*pi).*180./pi;

	% Apply manual unwraps
	if f0_rawplot == CMD.f0
		mask_manuw = (avgd_bias >= CMD.bias_low_V & avgd_bias <= CMD.bias_high_V);
		avgd_phase(mask_manuw) = avgd_phase(mask_manuw) + CMD.phase_correction;
	end


	%% Normalize to zero bias

	% Find phase at zero bias
	I_zb = (avgd_bias == 0);
	phase_zb = avgd_phase(I_zb);

	% Normalize points
	avgdn_phase = avgd_phase - phase_zb;

	f4=figure(4);
	f4.Position = [562 73 560 420];
	hold off;
	plot(avgd_bias, avg_phase, 'Marker', '.', 'LineStyle', ':',...
		'Color', [0, 0, 0.8], 'MarkerSize', 17);
	hold on;
	plot(avgd_bias, avgd_phase, 'Marker', 'o', 'LineStyle', ':',...
		'Color', [0.8, 0, 0], 'MarkerSize', 13);
	plot(avgd_bias, avgdn_phase, 'Marker', '+', 'LineStyle', ':',...
		'Color', [0, 0.65, 0], 'MarkerSize', 10, 'LineWidth', 1.3);
	xlabel("Bias Voltage (V)");
	ylabel("Phase (^\circ)");
	grid on;
	legend("Averaged", "Averaged + Unwrapped", "Averaged + Unwrapped + Normalized", 'Location', 'Best');


	%% Find q from phase

	len = 0.5; % meters
	Vp0 = 86.207e6; % m/s
	R_conv = 105; % Ohms




	% Limit bias
	if LIMIT_BIAS

		mask_limits = (abs(avgd_bias) >= bias_min_V) & (abs(avgd_bias) <= bias_max_V);

		avgdnm_phase = avgdn_phase(mask_limits);
		avgdnm_bias = avgd_bias(mask_limits);
	else
		avgdnm_phase = avgdn_phase;
		avgdnm_bias = avgd_bias;
	end

	% Remove zero point (don't divide by zero) and convert to current
	avgdnm_bias_nozero_A = avgdnm_bias(avgdnm_phase ~= 0)./R_conv;
	avgdnm_phase_nozero = avgdnm_phase(avgdnm_phase ~= 0);

	% Calculate q
	q = sqrt(180.*f0_rawplot.*len./abs(avgdnm_phase_nozero)./Vp0).*abs(avgdnm_bias_nozero_A);

	f5=figure(5);

	hold off;
	plot(avgdnm_bias_nozero_A.*1e3, q.*1e3, 'Marker', '.', 'LineStyle', ':',...
		'Color', [0, 0, 0.8], 'MarkerSize', 13);
	xlabel("Bias Current (mA)");
	ylabel("q (mA)");
	grid on;
	force0y;
	
	MASTER_DATA.(struct_key) = struct('bias_mA',avgdnm_bias_nozero_A.*1e3, 'q_mA', q.*1e3);
	
	% % Save to structs
	% q_vals.(struct_key) = q;
	% I_vals.(struct_key) = Ibias;
	% phase_vals.(struct_key) = cp;
end


%% Plot master data

q_avg = zeros(1, numel(keep_idxs));
q_std = zeros(1, numel(keep_idxs));

idx = 0;
for freq_idx = keep_idxs
	idx = idx + 1;
	
	struct_key = "Fidx"+num2str(freq_idx);
	
	q_avg(idx) = mean(MASTER_DATA.(struct_key).q_mA);
	q_std(idx) = std(MASTER_DATA.(struct_key).q_mA);
	
end

freqs = ds.configuration.frequency(keep_idxs);

figure(6);
hold off;
plot(freqs./1e9, q_avg, 'Marker', '.', 'LineStyle', ':', 'Color', [0.8, 0, 0], 'MarkerSize', 20);
hold on;
addPlotErrorbars(freqs./1e9, q_avg, q_std, false, 1, [0.8, 0, 0])
xlabel("Frequency (GHz)");
ylabel("q estimate (mA)");
grid on;



























%% Define functions



function addPlotErrorbars(Xdata, Ydata, Yerror, add_circ, circ_scaling, bar_color)
	
	if ~exist('bar_color', 'var')
		bar_color = [0.8, 0, 0];
	end
	
	% Scan over each point
	for idx = 1:numel(Yerror)
		line([Xdata(idx), Xdata(idx)], [Ydata(idx)-Yerror(idx), Ydata(idx)+Yerror(idx)], 'LineStyle', '-', 'Color', bar_color, 'LineWidth', 1.5, 'Marker', '_' , 'HandleVisibility', 'off');		
	end
	
	if add_circ
		scatter(Xdata, Ydata, Yerror.*circ_scaling, 'LineWidth', 1.5, 'MarkerEdgeColor', bar_color , 'HandleVisibility', 'off');
	end
end

function addPlotErrorbars2(Xdata, Ydata, YerrorL, YerrorH, add_circ, circ_scaling, bar_color)
	
	if ~exist('bar_color', 'var')
		bar_color = [0.8, 0, 0];
	end
	
	% Scan over each point
	for idx = 1:numel(YerrorL)
		line([Xdata(idx), Xdata(idx)], [YerrorL(idx), YerrorH(idx)], 'LineStyle', '-', 'Color', bar_color, 'LineWidth', 1.5, 'Marker', '_' , 'HandleVisibility', 'off');		
	end
	
	if add_circ
		scatter(Xdata, Ydata, abs(YerrorL+YerrorH)/2.*circ_scaling, 'LineWidth', 1.5, 'MarkerEdgeColor', bar_color , 'HandleVisibility', 'off');
	end
end






