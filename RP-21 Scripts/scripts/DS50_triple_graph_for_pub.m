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



CONVERT_V1 = false;


SKIP_READ = true;

CMAP_PWR = 'parula';
CMAP_FREQ = 'jet';
CMAP_CE = 'autumn';

%% Data Location

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
		
		if ispc
			DATA_PATH3 = "E:\ARC0 PhD Data\RP-21 Kinetic Inductance 2023\Data\group4_extflash\NIST September data";
		else
			DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
			DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
			DATA_PATH3 = fullfile('/', 'Volumes', 'NO NAME', 'NIST September data');
		end
		
		% v3.1
		filename = "gamma_9,87GHz_Target1_24Oct2023.mat";
		P_RF = 4; % dBm
		FREQ = 9.87e9; % Hz
		NORMAL_VOLTAGE = 0.001;
		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_1,5GHz_800MHzBW_20Oct2023.mat";
% 		P_RF = 3; % dBm
% 		FREQ = 1.3e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
% 		% v3.1
% 		filename = "gamma_10GHz_500MHzBW_20Oct2023.mat";
% 		P_RF = 5; % dBm
% 		FREQ = 9.9e9; % Hz
% 		NORMAL_VOLTAGE = 0.001;
% 		dpath = DATA_PATH3;
		
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





































