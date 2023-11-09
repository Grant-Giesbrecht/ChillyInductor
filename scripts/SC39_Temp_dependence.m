%% Configure data input

%----------------------------- FILE DATA|q -------------------------------

% Chip1 at 4,0K (all powers)
entry.temp = 4; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'BG 4K Sweeps', 'beta_fast2_4K0_30Oct2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'BG 4K Sweeps', 'gamma_9,87GHz_Target1_4K0_30Oct2023.mat');

% % Chip1 at 4,0K (all powers)
% entry.temp = 3; % K
% entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'BG 4K Sweeps', 'beta_fast2_4K0_30Oct2023.mat');
% entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'BG 4K Sweeps', 'gamma_9,87GHz_Target1_4K0_30Oct2023.mat');

%----------------------------- FILE DATA|LC -------------------------------



% Chip1 at 3,0K, 4 dBm
entry.temp = 3; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '3K', 'LC 3K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA = entry;

% Chip1 at 3,0K, -10 dBm
entry.temp = 3;
entry.P_RF = -10;
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '3K', 'LC 3K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2", "1V4", "1V6"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 4,0K 4 dBm
entry.temp = 4; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '4K', 'LC 4K 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 4,0K -10 dBm
entry.temp = 4; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '4K', 'LC 4K -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2", "1V4"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 5,0K 4 dBm
entry.temp = 5; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '5K', 'LC 5K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 5,0K -10 dBm
entry.temp = 5; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '5K', 'LC 5K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2", "1V4"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 6,0K 4 dBm
entry.temp = 6; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '6K', 'LC 6K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 6,0K -10 dBm
entry.temp = 6; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '6K', 'LC 6K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 7,0K 4 dBm
entry.temp = 7; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '7K', 'LC 7K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 7,0K -10 dBm
entry.temp = 7; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '7K', 'LC 7K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 8,0K 4 dBm
entry.temp = 8; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '8K', 'LC 8K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 8,0K -10 dBm
entry.temp = 8; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '8K', 'LC 8K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6, 0.8];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 9,0K 4 dBm
entry.temp = 9; % K
entry.P_RF = 4; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '9K', 'LC 9K0 4dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2"];
entry.bias_voltage = [0, 0.2];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

% Chip1 at 9,0K -10 dBm
entry.temp = 9; % K
entry.P_RF = -10; % dBm
entry.DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '9K', 'LC 9K0 -10dBm');
entry.FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6"];
entry.bias_voltage = [0, 0.2, 0.4, 0.6];
entry.EXTENSION = "_trimmed.s2p";
entry.S11_PREFIX = "LC_S11_";
entry.S21_PREFIX = "LC_S21_";
LC_ALL_DATA(end+1) = entry;

temp_list = unique([LC_ALL_DATA.temp]);
pwr_list = unique([LC_ALL_DATA.P_RF]);

% % Chip1 at 2,88K
% DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep','LC 2,88K -10dBm');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V1", "1V2", "1V4"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4];
% EXTENSION = ".prn";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

% % Chip2 at 4K
% DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Chip_r3c2_LC');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V1", "1V2"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2];
% EXTENSION = ".s2p";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

%% Plot L' and C' from ea. dataset (from SC27)

% Prepare plot styles
clr1 = [0, 0, 0.6];
clr2 = [0, 0.6, 0];
clr3 = [0.7, 0, 0];
marker_list = ['o', '*', '+', 'v', '^'];
mk_s = 'o';
mk_alt = '+';
standard_label = "No Loss Model";
alt_label = "With S_{21} Loss Model";

% Prepare analysis conditions struct
clear conditions;
conditions.NULL_NUM = 4; % Which null is being targeted
conditions.NULL_FREQ_LOW_HZ = 300e6; % Frequency bounds to target
conditions.NULL_FREQ_HIGH_HZ = 400e6;

% Reset figures
for fn = 1:3
	figure(fn);
	cla;
	hold off;
end

% Loop over each data entry
CM = resamplecmap('parula', numel(temp_list));
for e_idx = 1:numel(LC_ALL_DATA)
	
	
	% Get element
	entry = LC_ALL_DATA(e_idx);
	
	% Get indecies and markers
	temp_idx = find(entry.temp == temp_list);
	pwr_idx = find(entry.P_RF == pwr_list);
	
	dname = "P = "+num2str(entry.P_RF)+" dBm, T = "+num2str(entry.temp)+" K";
	
	% Lower and upper limits of bands in which to find detuned gamma magnitude
	conditions.DETUNE_FREQ_MIN_HZ = [100e6, 187e6, 276e6];
	conditions.DETUNE_FREQ_MAX_HZ = [156e6, 237e6, 328e6];
	
	[Ls, Cs] = LC_from_source(entry.DATA_PATH, entry.FILE_POSTFIXES, entry.EXTENSION, entry.S11_PREFIX, entry.S21_PREFIX, conditions);
	bias_voltage = entry.bias_voltage;
	Zcs = sqrt(Ls./Cs);
	
	mkz = 8;
	lw = 1.5;
	
	figure(1);
	plot(bias_voltage, Cs.*1e12, 'LineStyle', ':', 'Marker', mk_s, 'Color', CM(temp_idx, :), 'LineWidth', lw, 'MarkerSize', mkz, 'DisplayName', dname );
	hold on;
	grid on;
	xlabel("Bias Voltage (V)");
	ylabel("Distributed Capacitance (pF/m)");
	title("Distributed Capacitance over Bias");
	
% 	% fix limits
% 	C_avg = round(mean(Cs.*1e12));
% 	ylim([C_avg-1, C_avg+1]);
% 	yticks((C_avg-1):0.25:(C_avg+1));
	
% 	figure(2);
% 	plot(bias_voltage, Ls.*1e9, 'LineStyle', ':', 'Marker', mk_s, 'Color', CM(e_idx, :), 'LineWidth', lw, 'MarkerSize', mkz, 'DisplayName', dname );
% 	hold on;
% 	grid on;
% 	xlabel("Bias Voltage (V)");
% 	ylabel("Distributed Inductance (nH/m)");
% 	title("Distributed Inductance over Bias");
	
	figure(2);
	plot3(bias_voltage, ones(1, numel(bias_voltage)).*entry.temp , Ls.*1e9, 'LineStyle', ':', 'Marker', marker_list(pwr_idx), 'Color', CM(temp_idx, :), 'LineWidth', lw, 'MarkerSize', mkz, 'DisplayName', dname );
	hold on;
	grid on;
	xlabel("Bias Voltage (V)");
	ylabel("Temperature (K)");
	zlabel("Distributed Inductance (nH/m)");
	title("Distributed Inductance over Bias");
	
	figure(3);
	plot(bias_voltage, Zcs, 'LineStyle', ':', 'Marker', mk_s, 'Color', CM(temp_idx, :), 'LineWidth', lw, 'MarkerSize', mkz, 'DisplayName', dname );
	hold on;
	grid on;
	xlabel("Bias Voltage (V)");
	ylabel("Characteristic Impedance (Ohms)");
	title("Characteristic Impedance vs Bias");
	
end

for fn = 1:3
	figure(fn);
	legend('location', 'best');
end

%% Plot q versus temperature

warning off;

clear entry;

entry.temp = 3; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '3K', 'beta_fast2_3K0_31Oct2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '3K', 'gamma_9,87GHz_Target1_3K0_31Oct2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA = entry;

entry.temp = 4; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'beta_fast2_4K0_30Oct2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '4K', 'gamma_9,87GHz_Target1_4K0_30Oct2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

entry.temp = 5; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '5K', 'beta_fast2_5K0_1Nov2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '5K', 'gamma_9,87GHz_Target1_5K0_1Nov2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

entry.temp = 6; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '6K', 'beta_fast3_6K0_1Nov2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '6K', 'gamma_9,87GHz_Target1_6K0_1Nov2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

entry.temp = 7; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '7K', 'beta_fast3_7K0_2Nov2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '7K', 'gamma_9,87GHz_Target1_7K0_2Nov2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

entry.temp = 8; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '8K', 'beta_fast3_8K0_2Nov2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '8K', 'gamma_9,87GHz_Target1_8K0_2Nov2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

entry.temp = 9; % K
entry.BETA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '9K', 'beta_fast3_9K0_3Nov2023.mat');
entry.GAMMA_FN = fullfile('/', 'Volumes', 'NO NAME', 'October Temperature Sweep', '9K', 'gamma_9,87GHz_Target1_9K0_3Nov2023.mat');
load(entry.BETA_FN);
entry.ds = ds;
Q_ALL_DATA(end+1) = entry;

warning on;

%%

% Prepare conditions struct
clear conditions;
conditions.LIMIT_BIAS = true;
conditions.bias_max_A = 0.011;
conditions.bias_min_A = 0.005;
conditions.len = 0.5; % meters
conditions.Vp0 = 86.207e6; % m/s
conditions.iv_conv = 9.5e-3; % A/V

P_RF_ANALYZE = ds.configuration.RF_power(2);

% figure(4);
% subplot(1,1,1);
% hold off;
% 
% % This graph is plotted over frequency and different lines for different
% % temp. I would expect frequency and temperature to impact q, so it makes
% % sense that there is no easy to predict dependence.
% %
% % In the next graph I'll plot over constant (1.) frequency and (2.) temp so
% % that there should be a flat dependence.
% CMQ = resamplecmap('parula', numel(Q_ALL_DATA)+1);
% for e_idx = 1:numel(Q_ALL_DATA)
% 	
% 	% RUn analysis
% 	[Q_ALL_DATA(e_idx).mean_q_vs_freq, Q_ALL_DATA(e_idx).std_q_vs_freq, Q_ALL_DATA(e_idx).all_qs] = q_from_beta(Q_ALL_DATA(e_idx).ds, P_RF_ANALYZE, Q_ALL_DATA(e_idx).ds.configuration.frequency, conditions);
% 	
% 	dname = "P = "+num2str(P_RF_ANALYZE)+" dBm, T = "+num2str(Q_ALL_DATA(e_idx).temp)+" K";
% 	
% 	figure(4);
% 	plot(Q_ALL_DATA(e_idx).ds.configuration.frequency./1e9, Q_ALL_DATA(e_idx).mean_q_vs_freq.*1e3, 'LineStyle', ':', 'Marker', '+', 'Color', CMQ(e_idx, :), 'DisplayName', dname, 'LineWidth', 1, 'MarkerSize', 10);
% 	hold on;
% 	xlabel("Frequency (GHz");
% 	ylabel("q (mA)");
% 	title("q versus Frequency");
% 	grid on;
% 	
% end
% 
% figure(4);
% legend();
% xlim([9.855, 9.885]);

figure(5);
subplot(1,1,1);
hold off;

% Frequency to plot
FREQ_ANALYZE = 9.87e9;
temp_list = [Q_ALL_DATA.temp];
CMQ2 = resamplecmap('parula', numel(Q_ALL_DATA));
for e_idx = 1:numel(Q_ALL_DATA)
	
	entry = Q_ALL_DATA(e_idx);
	
	power_list = entry.ds.configuration.RF_power;
	
	skip_point = false;
	
	% Scan over all powers
	q_mean = zeros(1, numel(power_list));
	q_std = zeros(1, numel(power_list));
	for p_idx = 1:numel(power_list)
		
		try
			[q_mean(p_idx), q_std(p_idx), ~] = q_from_beta(entry.ds, power_list(p_idx), FREQ_ANALYZE, conditions);
		catch
			skip_point = true;
		end
		
		dname = "P = "+num2str(P_RF_ANALYZE)+" dBm, T = "+num2str(Q_ALL_DATA(e_idx).temp)+" K";
	
		
	end
	
	if skip_point
		continue;
	end
	
	temp_arr = ones(1, numel(power_list)).*entry.temp;
	
	figure(5);
	plot3(temp_arr, power_list, q_mean.*1e3, 'LineStyle', ':', 'Marker', '+', 'Color', CMQ2(e_idx, :), 'DisplayName', dname, 'LineWidth', 2, 'MarkerSize', 10);
	hold on;
	addPlot3Errorbars(temp_arr, power_list, q_mean.*1e3, q_std.*1e3, true, 0.5);
	xlabel("Temperature (K)");
	ylabel("RF Power (dBm)");
	zlabel("q (mA)");
	title("q versus Frequency");
	grid on;
	
	Q_ALL_DATA(e_idx) = entry;
end

figure(5);
legend('location', 'best');

function addPlot3Errorbars(Xdata, Ydata, Zdata, Zerror, add_circ, circ_scaling)
	
	% Scan over each point
	for idx = 1:numel(Zerror)
		line([Xdata(idx), Xdata(idx)], [Ydata(idx), Ydata(idx)], [Zdata(idx)-Zerror(idx), Zdata(idx)+Zerror(idx)], 'LineStyle', '-', 'Color', [0.8, 0, 0], 'LineWidth', 1.5, 'Marker', '_' , 'HandleVisibility', 'off');		
	end
	
	if add_circ
		scatter3(Xdata, Ydata, Zdata, Zerror.*circ_scaling, 'LineWidth', 1.5, 'MarkerEdgeColor', [0.8, 0, 0] , 'HandleVisibility', 'off');
	end
end












