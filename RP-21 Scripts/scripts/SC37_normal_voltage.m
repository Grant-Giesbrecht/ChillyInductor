%% Select datafile

FIG_NUM = 1;

DATA_PATH1 = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');
DATA_PATH3 = fullfile('/', 'Volumes', 'NO NAME', 'NIST September data');

% v3.1
filename = "gamma_1,5GHz_800MHzBW_20Oct2023.mat";
P_RF = 3; % dBm
FREQ = 1.3e9; % Hz
NORMAL_VOLTAGE = 0.001;
dpath = DATA_PATH3;

% % v3.1
% filename = "gamma_10GHz_500MHzBW_20Oct2023.mat";
% P_RF = 5; % dBm
% FREQ = 9.9e9; % Hz
% NORMAL_VOLTAGE = 0.001;
% dpath = DATA_PATH3;

% % v3.1
% filename = "gamma_9,7GHz_200MHzBW_19Oct2023.mat";
% P_RF = 5; % dBm
% FREQ = 9.65e9; % Hz
% NORMAL_VOLTAGE = 0.001;
% dpath = DATA_PATH3;


% % v3.1
% filename = "gamma_10GHz_1GHzBW_19Oct2023.mat";
% P_RF = 5; % dBm
% FREQ = 10e9; % Hz
% NORMAL_VOLTAGE = 0.001;
% dpath = DATA_PATH3;

% % Best 4K Data (Chip 2)
% filename = 'gamma_14Sept2023_1600.mat';
% P_RF = 3; % dBm
% FREQ = 10e9; % Hz
% NORMAL_VOLTAGE = 0.002;
% dpath = DATA_PATH1;

if ~exist('ds', 'var')
	displ("Reading file", fullfile(dpath, filename));
	load(fullfile(dpath, filename));
end

%% Find normal voltage vs frequency

% Generate local conditions struct
c = defaultConditions();

c.convert_to_W = 1;
c.Vnorm = NORMAL_VOLTAGE;

V_max_SC = zeros(numel(ds.configuration.frequency), numel(ds.configuration.RF_power));

figure(FIG_NUM);
hold off;

CM = resamplecmap('parula', numel(ds.configuration.RF_power));

mkz = 20;
lw = 1;

% Scan over all power levels
for pidx = 1:numel(ds.configuration.RF_power)
	
	power = ds.configuration.RF_power(pidx);
	c.SG_power = power;
	
	% Scan over all frequencies
	for fidx = 1:numel(ds.configuration.frequency)
		
		freq = ds.configuration.frequency(fidx);
		
		% Update conditions
		c.f0 = freq;

		% Extract data
		[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
		
		% Find last voltage that was not normal - save
		In = (norm.V < NORMAL_VOLTAGE);
		[~, Idx_max] = max(norm.V(In));
		V_max_SC(fidx, pidx) = ds.configuration.bias_V(Idx_max);
	end
	
	figure(FIG_NUM);
	plot(ds.configuration.frequency./1e9, V_max_SC(:, pidx), 'Color', CM(pidx, :), 'LineStyle', '--', 'Marker', '.', 'DisplayName', strcat("P_{RF} = ", num2str(power), " dBm"), 'LineWidth', lw, 'MarkerSize', mkz);
	hold on;
	
	displ(" --> ", pidx/numel(ds.configuration.RF_power)*100, "%");
	
end

figure(FIG_NUM);
xlabel("Frequency (GHz)");
ylabel("Maximum SC Bias (V)");
title("Bias Voltage Frequency Dependence");
grid on;
force0y;

%% Calculate and plot averages (across power)

avg_Vmax = zeros(1, numel(ds.configuration.RF_power));
for pidx = 1:numel(ds.configuration.RF_power)
	avg_Vmax(pidx) = mean(V_max_SC(:, pidx));
end

figure(FIG_NUM+1);
hold off
plot(ds.configuration.RF_power, avg_Vmax, 'LineStyle', '--', 'Marker', '.', 'MarkerSize', mkz, 'LineWidth', lw, 'Color', [0, 0, 0.6]);
xlabel("P_{RF} (dBm)");
ylabel("Mean Max Bias (V)");
grid on;
force0y;
title("Power Trend, FILE="+strrep(filename, '_', '\_'));

%% Calculate and plot averages (across frequency)

avg_Vmax_f = zeros(1, numel(ds.configuration.frequency));
for fidx = 1:numel(ds.configuration.frequency)
	avg_Vmax_f(fidx) = mean(V_max_SC(fidx, :));
end

figure(FIG_NUM+2);
hold off
plot(ds.configuration.frequency./1e9, avg_Vmax_f, 'LineStyle', '--', 'Marker', '.', 'MarkerSize', mkz, 'LineWidth', lw, 'Color', [0, 0, 0.6]);
xlabel("Frequency (GHz)");
ylabel("Mean Max Bias (V)");
grid on;
force0y;
title("Frequency Trend, FILE="+strrep(filename, '_', '\_'));























