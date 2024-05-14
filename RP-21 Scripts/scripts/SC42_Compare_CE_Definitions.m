%%

t0 = tic();

%% Configuration Data

SKIP_READ = true;

CMAP_PWR = 'parula';
CMAP_FREQ = 'jet';
CMAP_CE = 'autumn';

%% Data Location

if ~SKIP_READ || ~exist('ds', 'var')

	DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
	DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
% 	DATA_PATH3 = fullfile('/', 'Volumes', 'NO NAME', 'NIST September data');
	DATA_PATH3 = '/Volumes/M4 PHD/ARC0 PhD Data/RP-21 Kinetic Inductance 2023/Data/group4_extflash/NIST September data';

	% v3.1
	filename = "gamma_9,87GHz_Target1_24Oct2023.mat";
	P_RF = 4; % dBm
	FREQ = 9.87e9; % Hz
	NORMAL_VOLTAGE = 0.001;
	dpath = DATA_PATH3;

% 	% v3.1
% 	filename = "gamma_1,5GHz_800MHzBW_20Oct2023.mat";
% 	P_RF = 3; % dBm
% 	FREQ = 1.3e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1
% 	filename = "gamma_10GHz_500MHzBW_20Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 9.9e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1
% 	filename = "gamma_9,7GHz_200MHzBW_19Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 9.65e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1
% 	filename = "gamma_10GHz_1GHzBW_19Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 10e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1
% 	filename = "gamma_13,3GHz_40MHzBW_v3,1_18Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 13.34e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1
% 	filename = "gamma_9,7GHz_200MHzBW_v3,1_18Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 9.65e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% v3.1?
% 	filename = "gamma_9,615GHz_20MHzBW_16Oct2023.mat";
% 	P_RF = 4.5; % dBm
% 	FREQ = 9.615e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% 
% 	filename = "gamma_13,3GHz_40MHzBW_13Oct2023.mat";
% 	P_RF = 4.5; % dBm
% 	FREQ = 13.34e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% 2.8K (Chip 1, m2) 13.9-14.1 GHz, 20 MHz steps
% 	filename = "gamma_14GHz_200MHzBW_13Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 14e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% 2.8K (Chip 1, m2) 9.6-9.7 GHz, 12.5 MHz steps
% 	filename = "gamma_9,7GHz_200MHzBW_13Oct2023.mat";
% 	P_RF = 5; % dBm
% 	FREQ = 9.65e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% Best 2.8K data, Narrow (Chip 1, m2) 13.3 GHz ballpark (0.25 GHz
% 	% spread)
% 	filename = "gamma_narrow3_11Oct2023.mat";
% 	P_RF = 4; % dBm
% 	FREQ = 13.3e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% Best 2.8K data, Narrow (Chip 1, m2) 10 GHz
% 	filename = "gamma_narrow1_10GHz_11Oct2023.mat";
% 	P_RF = 4; % dBm
% 	FREQ = 10e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 
% 	% Narrow(er/2) 2.8K data (Chip 1, m2) (13-13.8 GHz, 50 MHz steps)
% 	filename = "gamma3Knarrow2_9Oct2023.mat";
% 	P_RF = 4; % dBm
% 	FREQ = 13.2e9; % Hz
% 	NORMAL_VOLTAGE = 1e-3;
% 	dpath = DATA_PATH3;
% 
% 	% Narrow 2.8K data (Chip 1, m2) (12-14 GHz?, 200 MHz steps?)
% 	filename = "gamma_narrow_8Oct2023.mat";
% 	P_RF = 3; % dBm
% 	FREQ = 13e9; % Hz
% 	NORMAL_VOLTAGE = 1e-3;
% 	dpath = DATA_PATH3;
% 
% 	% Best 2.8K data, Wide (Chip 1, m2) (1-15 GHz, 1 GHz steps)
% 	filename = 'gamma3Kfine_3Oct2023.mat';
% 	P_RF = 0; % dBm
% 	FREQ = 10e9; % Hz
% 	NORMAL_VOLTAGE = 0.001;
% 	dpath = DATA_PATH3;
% 
% 	% Best 4K Data (Chip 2)
% 	filename = 'gamma_14Sept2023_1600.mat';
% 	P_RF = 3; % dBm
% 	FREQ = 10e9; % Hz
% 	NORMAL_VOLTAGE = 0.002;
% 	dpath = DATA_PATH1;
% 
% 	% Best 3K Data (Chip 2)
% 	filename = 'gamma3K_21Sept2023_3K.mat';
% 	P_RF = 3; % dBm
% 	FREQ = 10e9; % Hz
% 	NORMAL_VOLTAGE = 0.00075;
% 	dpath = DATA_PATH1;

	displ("Reading file", fullfile(dpath, filename));
	load(fullfile(dpath, filename));
end
% Read Config data
%

powers_dBm = ds.configuration.RF_power;
freqs = ds.configuration.frequency;

% Generate default settings
ah_fig_set = struct('P_RF', P_RF, 'FREQ', FREQ, 'NORMAL_VOLTAGE', NORMAL_VOLTAGE, 'CMAP', 'parula');

s2p_dpath = '/Volumes/M5 PERSONAL/Warm_Cal_S21/';
file_cryocal = fullfile(s2p_dpath, "SP_50MHz_50GHz_cryo2_trim.s2p");
s2p_data = sparameters(file_cryocal);

%% Run analysis - Figure 1

% Generate figure
[fig1, data1] = analyze_harmonics(ds, 'max_ce_vs_freq_power', ah_fig_set, ...
	'Fig', 1, 'StatusUpdates', true, 'CEDefinition', 'Chip');

%% Run analysis - Figure 2

% Generate figure
[fig2, data2] = analyze_harmonics(ds, 'max_ce_vs_freq_power', ah_fig_set, ...
	'Fig', 2, 'StatusUpdates', true, 'CEDefinition', 'System');

%% Run analysis - Figure 3

% Generate figure
[fig3, data3] = analyze_harmonics(ds, 'max_ce_vs_freq_power', ah_fig_set, ...
	'Fig', 3, 'StatusUpdates', true, 'CEDefinition', 'SystemS11', 'S2PData', s2p_data);