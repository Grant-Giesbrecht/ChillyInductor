%% Configuration Data

P_RF = 2; % dBm
FREQ = 10e9; % Hz

CONVERT_V1 = false;

% NORMAL_VOLTAGE = 0.00035;
NORMAL_VOLTAGE = 0.002;

SKIP_READ = true;

CMAP_PWR = 'parula';
CMAP_FREQ = 'jet';
CMAP_CE = 'autumn';

%% Data Location

if CONVERT_V1
	
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
	
else
	if ~SKIP_READ || ~exist('ds', 'var')
		DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
		DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
		DATA_PATH3 = fullfile('G:', 'NIST September data');

		filename = 'gamma_14Sept2023_1600.mat';
% 		filename = 'gamma3K_21Sept2023_3K.mat';

		load(fullfile(DATA_PATH1, filename));
	end
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
ah_fig_set.CMAP = 'jet';
fig3 = analyze_harmonics(ds, 'harm_power_vs_freq', ah_fig_set, 'Fig', 13);

%% Run analysis - Figure 4

% Generate figure
ah_fig_set.CMAP = 'autumn';
fig4 = analyze_harmonics(ds, 'max_ce_vs_freq_power', ah_fig_set, 'Fig', 14);

%% Run analysis - Figure 5

% Generate figure
ah_fig_set.CMAP = 'autumn';
fig5 = analyze_harmonics(ds, 'ce_vs_bias_power', ah_fig_set, 'Fig', 15);








































