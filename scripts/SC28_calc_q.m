%% SC28
%
% Accepts a beta sweep and plots q estimates, tables, distribution etc.
%
% Previously named DS51
%
% Built off of SC1

DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
DATA_PATH_SUPP = fullfile('/','Volumes','NO NAME', 'NIST Datasets Supplementary', 'group3_supplementary');

% filename = 'beta_12Sept2023.mat'; % This is the file with the disconnected bias cable and thus flat response.
% filename = 'beta_14Sept2023_1600.mat'; % Best 4K Chip(3-2) dataset
% analysis_freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].*1e9;

% filename = 'beta3K_20Sept2023_3K.mat'; % Best 3K Chip(3-2) dataset
% load(fullfile(DATA_PATH, filename));
% analysis_freqs = (1:20).*1e9;

% filename = "gamma_14Sept2023_1600.mat";
% load(fullfile(DATA_PATH2, filename));
% analysis_freqs = ds.configuration.frequency;

% filename = 'betaV3_29Sept2023_autosave.mat'; % This is a partial Chip-1 2.8K measurement (failed when laptop slept).
% load(fullfile(DATA_PATH_SUPP, filename));
% analysis_freqs = [1, 2, 3, 4, 5, 7, 9, 10].*1e9;

%%-- Convert V2 Data format to expected format for SC1

P_RF = 0;

% analysis_freqs = [1, 5, 10, 15].*1e9; % Frequencies to plot [GHz]

% load(fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023', 'Data', 'group4_extflash', 'October Temperature Sweep', '4K', 'beta_fast2_4K0_30Oct2023.mat'));
% load(fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023', 'Data', 'group4_extflash', 'December data', 'beta_onelambda_V1_05Dec2023.mat'));
load(fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023', 'Data', 'group4_extflash', 'December data', 'beta_10GHz_V1_07Dec2023.mat'));

% analysis_freqs = (1:1:8).*1e9;
% invmask = (analysis_freqs == 4e9);
% analysis_freqs = analysis_freqs(~invmask);


% analysis_freqs = ds.configuration.frequency(5:end-7);
analysis_freqs = ds.configuration.frequency;
% analysis_freqs = ds.configuration.frequency(10);

LIMIT_BIAS = false;

bias_max_A = 0.011;
bias_min_A = .005;
% bias_min_A = 0;

%% View Dataset

ZOOM_ONE_FREQ = true;

figure(20);
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

%% Resume Analysis

%%-----------------------------

% This script is built-off phase_to_x but uses correctData().
% Previously named: VNA_fundamental2_30March.m
%
% Analyzes phase data with I^2/q^2 model
%
% Note: this datasweep is looking at how the 

len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

CM = resamplecmap('parula', numel(analysis_freqs));

% Process each frequency
for f = analysis_freqs
	
	struct_key = string(strcat('f', num2str(f/1e6)));
	
	% Apply drift correction, unwrap phase, and normalize
	[cp, vb] = getCorrectedPhase_V2(ds, P_RF, f, false);
	
	% Get bias currents
	Ibias = iv_conv.*vb;
	
	% Limit bias
	if LIMIT_BIAS
		
		Ipass = (abs(Ibias) >= bias_min_A) & (abs(Ibias) <= bias_max_A);
		cp = cp(Ipass);
		vb = vb(Ipass);
		Ibias = Ibias(Ipass);
		
	end
	
	% Remove zero point (don't divide by zero)
	I_calc = Ibias(cp ~= 0);
	cp_calc = cp(cp ~= 0);
	
	% Calculate q
	q = sqrt(180.*f.*len./abs(cp_calc)./Vp0).*abs(I_calc);
	
	% Save to structs
	q_vals.(struct_key) = q;
	I_vals.(struct_key) = Ibias;
	phase_vals.(struct_key) = cp;
end

%% Show summary

lw = 1.5;
mkz = 10;

figure(1);
hold off;

figure(2);
hold off;

mt = MTable();
mt.title("Nonlinearity Summary");
mt.row(["Freq (GHz)", "avg(q) [mA]", "min(q) [mA]", "max(q) [mA]", "stdev(q) [mA]"]);

legend_vals = {};

all_qs = [];
avg_q = [];
std_q  = [];

% Process each frequency
idx = 0;
for f = analysis_freqs
	idx = idx + 1;

	% Get key
	struct_key = string(strcat('f', num2str(f/1e6)));
	
	% Get data
	q = q_vals.(struct_key);
	Ibias = I_vals.(struct_key);
	phase = phase_vals.(struct_key);
	
	% Add all qs
	all_qs = [all_qs, q];
	avg_q = [avg_q, mean(q)];
	std_q = [std_q, std(q)];
	
	% Add to graphs
	figure(1);
	plot(Ibias, phase, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz, 'Color', CM(idx, :));
	hold on;
	
	figure(2);
	plot(Ibias(phase ~= 0), q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz, 'Color', CM(idx, :));
	hold on;
	
	% Add to table
	mt.row([string(num2str(f/1e9)), string(num2str(mean(q.*1e3))), string(num2str(min(q.*1e3))), string(num2str(max(q.*1e3))), string(num2str(std(q.*1e3))) ]);
	
	% Add to legend
	legend_vals = [legend_vals(:)', {strcat(num2str(f/1e9), " GHz")}];
end

all_q_mean = mean(all_qs)*1e3;
all_q_std = std(all_qs)*1e3;

% Print table
disp(mt.str());
displ(newline, "Total Average: ", all_q_mean, " mA");
displ("Total St.Dev.: ", all_q_std, " mA");

% Finish graphs
figure(1);
grid on;
legend(legend_vals{:});
title("Phase Change from Zero-Bias");
xlabel("Bias Current (mA)");
ylabel("\Delta Phase (deg)");

c_std = [1, 1, 1].*0.3;
c_mean = [0, 0, 0];
figure(2);
grid on;
legend(legend_vals{:});
title("Nonlinearity Estimate");
xlabel("Bias Current (mA)");
ylabel("q (A)");
hlin(all_q_mean/1e3, 'LineStyle', '--', 'Color', c_mean);
hlin((all_q_mean+all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);
hlin((all_q_mean-all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);

% Plot over frequency
figure(3);
hold off;
plot(analysis_freqs./1e9, avg_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
hold on;
plot(analysis_freqs./1e9, std_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Frequency (GHz)");
ylabel("q (A)");
title("Nonlinearity over frequency");
legend("Mean", "Standard Deviation");

% Plot over frequency (error bars)

figure(5);
if LIMIT_BIAS
	clr = [0.8, 0.2, 0.2];
else
	clr = [0.2, 0.2, 0.8];
	hold off;
end
plot(analysis_freqs./1e9, avg_q.*1e3, 'LineStyle', ':', 'LineWidth', lw, 'Color', clr);
hold on;
addPlotErrorbars(analysis_freqs./1e9, avg_q.*1e3, std_q.*1e3, true, 10, clr );
grid on;
xlabel("Frequency (GHz)");
ylabel("q (A)");
title("Nonlinearity over frequency");
legend("Mean", "Standard Deviation");

% Plot histogram
sel_qs = all_qs(all_qs < 0.4);
figure(4);
hold off;
histogram(sel_qs, 20, 'FaceColor', [0, 0, .7]);
hold on;
vlin(mean(sel_qs), "LineStyle", '--', 'Color', c_mean);
vlin(mean(sel_qs)-std(sel_qs), "LineStyle", ':', 'Color', c_std);
vlin(mean(sel_qs)+std(sel_qs), "LineStyle", ':', 'Color', c_std);
xlabel("q Value (A)");
ylabel("Counts");
title("Distribution of q Estimates");


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