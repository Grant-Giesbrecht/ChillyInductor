%% SC30 Harmonic Analysis V2
%
% Built off of SC25 and SC26. Performs data analysis on v2 data,
% calculating conversion efficiency, etc.
%
% Run time:
%	* 550 sec on P15Gen1 (for) though fig 4
%
% NOTE: SC31 is now recommended, along with analyze_harmonics(). SC31
% performs an identical analysis to SC30, however it uses a function which
% can more easily be expanded to future scripts.

t0 = tic();

%% Configuration Data

<<<<<<< HEAD:RP-21 Scripts/scripts/SC30_harmonic_analysis_v2.m
P_RF = 4; % dBm
FREQ = 1.5e9; % Hz

NORMAL_VOLTAGE = 0.001;
=======


CONVERT_V1 = false;

>>>>>>> 2c06138f2d8d06ec776477c10e92b4fb352379a8:scripts/SC30_harmonic_analysis_v2.m

SKIP_READ = true;

CMAP_PWR = 'parula';
CMAP_FREQ = 'jet';
CMAP_CE = 'autumn';

%% Data Location

<<<<<<< HEAD:RP-21 Scripts/scripts/SC30_harmonic_analysis_v2.m
if ~SKIP_READ || ~exist('ds', 'var')
	DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
	DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
	DATA_PATH3 = fullfile('D:', 'NIST September data');
% 	DATA_PATH3 = fullfile('C:','Users','Grant Giesbrecht', 'Downloads');
	DATA_PATH4 = 'C:\Users\Grant Giesbrecht\Desktop\pymuf\scripts_grant\universal\datasets';

	filename = 'gamma_14Sept2023_1600.mat';
	filename = 'gamma3Kfine_3Oct2023.mat';
	filename = 'gamma3Kfine_3Oct2023.mat';
	filename = "gamma_narrow3_11Oct2023.mat";
	filename = "gamma_1,5GHz_800MHzBW_20Oct2023_autosave.mat";
	filename = "gamma_1,5GHz_800MHzBW_20Oct2023.mat";

	load(fullfile(DATA_PATH3, filename));
=======
if CONVERT_V1
	
	displ("Reading V1 data");
	
	% Load V1 Data
	load(dataset_path("DS5_FinePower_PO-1.mat"));
	load(dataset_path("cryostat_sparams.mat"));
	
	% Create V2 proxy data
	clear ds;
	ds.dataset = ld;
	ds.configuration.RF_power = unique([ld.SG_power_dBm]);
	ds.configuration.frequency = unique([ld.SG_freq_Hz]);
	for idx = 1:numel(ds.dataset)
		ds.dataset(idx).('temp_K') = NaN;
	end
	
	warning("This does not consider system loss for V1 and thus won't be as holistic an indication an SC25 & SC26!");
	
% 	% Analyze V1 system loss data
% 
% 	idx_h1 = findClosest(freq_Hz, FREQ);
% 	idx_h2 = findClosest(freq_Hz, FREQ*2);
% 	idx_h3 = findClosest(freq_Hz, FREQ*3);
% 
% 	S21_h1 = S21_dB(idx_h1);
% 	S21_h2 = S21_dB(idx_h2);
% 	S21_h3 = S21_dB(idx_h3);
	
else
	if ~SKIP_READ || ~exist('ds', 'var')
		
		DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
		DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
% 		DATA_PATH3 = fullfile('/', 'Volumes', 'NO NAME', 'NIST September data');
		DATA_PATH3 = fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023', 'Data', 'group4_extflash', 'NIST September data');
		
% 		% v3.1
% 		filename = "gamma_9,87GHz_Target1_24Oct2023.mat";
% 		P_RF = 4; % dBm
% 		FREQ = 9.87e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_1,5GHz_800MHzBW_20Oct2023.mat";
% 		P_RF = 3; % dBm
% 		FREQ = 1.3e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
		% v3.1
		filename = "gamma_10GHz_500MHzBW_20Oct2023.mat";
		P_RF = 5; % dBm
		FREQ = 9.9e9; % Hz
		NORMAL_VOLTAGE = 0.001;
		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_9,7GHz_200MHzBW_19Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 9.65e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_10GHz_1GHzBW_19Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 10e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_13,3GHz_40MHzBW_v3,1_18Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 13.34e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_9,7GHz_200MHzBW_v3,1_18Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 9.65e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1?
% 		filename = "gamma_9,615GHz_20MHzBW_16Oct2023.mat";
% 		P_RF = 4.5; % dBm
% 		FREQ = 9.615e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% 
% 		filename = "gamma_13,3GHz_40MHzBW_13Oct2023.mat";
% 		P_RF = 4.5; % dBm
% 		FREQ = 13.34e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% 2.8K (Chip 1, m2) 13.9-14.1 GHz, 20 MHz steps
% 		filename = "gamma_14GHz_200MHzBW_13Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 14e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% 2.8K (Chip 1, m2) 9.6-9.7 GHz, 12.5 MHz steps
% 		filename = "gamma_9,7GHz_200MHzBW_13Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 9.65e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% Best 2.8K data, Narrow (Chip 1, m2) 13.3 GHz ballpark (0.25 GHz
% 		% spread)
% 		filename = "gamma_narrow3_11Oct2023.mat";
% 		P_RF = 4; % dBm
% 		FREQ = 13.3e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% Best 2.8K data, Narrow (Chip 1, m2) 10 GHz
% 		filename = "gamma_narrow1_10GHz_11Oct2023.mat";
% 		P_RF = 4; % dBm
% 		FREQ = 10e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;

		
% 		% Narrow(er/2) 2.8K data (Chip 1, m2) (13-13.8 GHz, 50 MHz steps)
% 		filename = "gamma3Knarrow2_9Oct2023.mat";
% 		P_RF = 4; % dBm
% 		FREQ = 13.2e9; % Hz
% 		NORMAL_VOLTAGE = 1e-3;
% 		dpath = DATA_PATH3;
		
% 		% Narrow 2.8K data (Chip 1, m2) (12-14 GHz?, 200 MHz steps?)
% 		filename = "gamma_narrow_8Oct2023.mat";
% 		P_RF = 3; % dBm
% 		FREQ = 13e9; % Hz
% 		NORMAL_VOLTAGE = 1e-3;
% 		dpath = DATA_PATH3;
		
% 		% Best 2.8K data, Wide (Chip 1, m2) (1-15 GHz, 1 GHz steps)
% 		filename = 'gamma3Kfine_3Oct2023.mat';
% 		P_RF = 0; % dBm
% 		FREQ = 10e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% Best 4K Data (Chip 2)
% 		filename = 'gamma_14Sept2023_1600.mat';
% 		P_RF = 3; % dBm
% 		FREQ = 10e9; % Hz
% 		NORMAL_VOLTAGE = 0.002;
% 		dpath = DATA_PATH1;
		
% 		% Best 3K Data (Chip 2)
% 		filename = 'gamma3K_21Sept2023_3K.mat';
% 		P_RF = 3; % dBm
% 		FREQ = 10e9; % Hz
% 		NORMAL_VOLTAGE = 0.00075;
% 		dpath = DATA_PATH1;
		
		displ("Reading file", fullfile(dpath, filename));
		load(fullfile(dpath, filename));
	end
>>>>>>> 2c06138f2d8d06ec776477c10e92b4fb352379a8:scripts/SC30_harmonic_analysis_v2.m
end
% Read Config data
%

powers_dBm = ds.configuration.RF_power;
freqs = ds.configuration.frequency;

%% Run analysis - Figure 1

% Generate default settings
ah_fig_set = struct('P_RF', P_RF, 'FREQ', FREQ, 'NORMAL_VOLTAGE', NORMAL_VOLTAGE, 'CMAP', 'parula');

% Generate figure
fig1 = analyze_harmonics(ds, 'harm_power', ah_fig_set, 'Fig', 11);

%% Run analysis - Figure 2

% Generate figure
fig2 = analyze_harmonics(ds, 'harm_power_vs_pin', ah_fig_set, 'Fig', 12);

%% Run analysis - Figure 3

% Generate figure
fig3 = analyze_harmonics(ds, 'harm_power_vs_freq', ah_fig_set, 'Fig', 13);

%% Run analysis - Figure 4

% Generate figure
[fig4, data4] = analyze_harmonics(ds, 'max_ce_vs_freq_power', ah_fig_set, 'Fig', 4, 'StatusUpdates', true, 'CEDefinition', 'Chip');

%% Plot biases from fig 4

figure(5);
hold off;
keys = ccell2mat(fields(data4));
NK = numel(keys);
CM = resamplecmap('parula', NK+1);
ALL_BIASES = []; % in mA
for idx = 1:NK
	plot(flatten(ds.configuration.frequency./1e9), flatten(data4.(keys(idx)).Vbias_CE2_chip./105.*1e3), 'Marker', '.', 'LineStyle', ':', 'Color', CM(idx, :), 'DisplayName', keys(idx), 'LineWidth', 1.3, 'MarkerSize', 20);
	hold on;
	ALL_BIASES = [ALL_BIASES, flatten(data4.(keys(idx)).Vbias_CE2_chip./105.*1e3)];
end
grid on;
xlabel("Frequency (GHz)");
ylabel("Optimum Bias (mA)");
legend('Location', "Best");
force0y;

displ("Bias Range: ", min(ALL_BIASES), " -> ", max(ALL_BIASES), " mA");

%% Run analysis - Figure 5

% Generate figure
fig5 = analyze_harmonics(ds, 'ce_vs_bias_power', ah_fig_set, 'Fig', 15);

%% Run analysis - Figure 6

% Generate figure
ah_fig_set.CMAP = 'parula';
fig6 = analyze_harmonics(ds, 'vmfli_vs_bias', ah_fig_set, 'Fig', 6, 'Hold', false);

%% Run analysis - Figure 7

% Generate figure
data_freqs = unique([ds.dataset.SG_freq_Hz]);
ah_fig_set.FREQ = data_freqs(2:7);

fig6 = analyze_harmonics(ds, 'ce2_Vs_bias_power_freq', ah_fig_set, 'Fig', 7, 'EqualizeScales', true);

%% Report elapsed time

tf = tic();

t_elapsed = toc(t0)-toc(tf);
mins = floor(t_elapsed/60);
sec = t_elapsed - 60*mins;
displ("Analysis completed in ", mins, " minutes, ", round(sec*10)/10, " seconds");






































