DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
DATA_PATH3 = fullfile('/', 'Volumes', 'NO NAME', 'NIST September data');
DATA_PATH3 = "E:\ARC0 PhD Data\RP-21 Kinetic Inductance 2023\Data\group4_extflash\NIST September data";

% v3.1
filename = "gamma_10GHz_1GHzBW_19Oct2023.mat";
P_RF = 5; % dBm
FREQ = 10e9; % Hz
NORMAL_VOLTAGE = 0.001;
dpath = DATA_PATH3;

load(fullfile(dpath, filename));

%% Generate const RF_power surface plot

RF_power = 4; % dBm

% Generate default settings
ah_fig_set = struct('P_RF', RF_power, 'FREQ', 10e9, 'NORMAL_VOLTAGE', 1e-3, 'CMAP', 'parula');

% Generate figure
fig1 = analyze_harmonics(ds, 'CE_surf_freq_vs_bias', ah_fig_set, 'Fig', 1, 'EqualizeScales', true);