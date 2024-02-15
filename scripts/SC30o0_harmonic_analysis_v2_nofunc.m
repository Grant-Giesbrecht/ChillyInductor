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
		DATA_PATH3 = fullfile('G:', 'NIST September data');
		
		filename = 'gamma_14Sept2023_1600.mat';
% 		filename = 'gamma3K_21Sept2023_3K.mat';
		
		displ("Reading file", fullfile(DATA_PATH1, filename));
		load(fullfile(DATA_PATH1, filename));
	end
end
% Read Config data
%

powers_dBm = ds.configuration.RF_power;
freqs = ds.configuration.frequency;

%% Extract data for Selected Single Condition

c = defaultConditions();
c.SG_power = P_RF;
c.convert_to_W = 1;
c.f0 = FREQ;
c.Vnorm = NORMAL_VOLTAGE;

charm1 = [0, 0, 0.7];
charm2 = [0, 0.6, 0];
charm3 = [0.6, 0, 0];

% function [harm_struct, normal, Vsweep] = getHarmonicSweep_v2(rich_data, c, keep_normal)
[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);

figure(1);
subplot(1, 1, 1);
hold off;
plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', charm1);
hold on;
plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', charm2);
plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', charm3);
grid on;
xlabel("Bias Voltage (V)");
ylabel("Harmonic Power at Chip Output (dBm)");
title("Freq = " + num2str(FREQ./1e9) + " GHz, P = " + num2str(P_RF) + " dBm");

%% Plot each harmonic in power sweep (at selected frequency)

figure(2);
subplot(1, 3, 1);
hold off;
subplot(1, 3, 2);
hold off;
subplot(1, 3, 3);
hold off;

LL4 = {};
CM = resamplecmap(CMAP_PWR, numel(powers_dBm));
idx = 0;
for pwr = powers_dBm
	idx = idx + 1;
	
	% Update conditions
	c.f0 = FREQ;
	c.SG_power = pwr;
	
	% Extract data
	[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
	
	figure(2);
	subplot(1, 3, 1);
	plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	subplot(1, 3, 2);
	plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	subplot(1, 3, 3);
	plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	
	LL4 = [LL4(:)', {"P = "+num2str(pwr) + " dBm"}];
	
end

figure(2);
subplot(1, 3, 1);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("Fundamental's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
grid on;
subplot(1, 3, 2);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("2nd Harmonic's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
grid on;
subplot(1, 3, 3);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("3rd Harmonic's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
grid on;


%% Plot each harmonic in power sweep (at selected frequency)

figure(3);
subplot(1, 3, 1);
hold off;
subplot(1, 3, 2);
hold off;
subplot(1, 3, 3);
hold off;

LL4 = {};
CM = resamplecmap(CMAP_FREQ, numel(freqs));
idx = 0;
for f = freqs
	idx = idx + 1;
	
	% Update conditions
	c.f0 = f;
	c.SG_power = P_RF;
	
	% Extract data
	[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
	
	figure(3);
	subplot(1, 3, 1);
	plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	subplot(1, 3, 2);
	plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	subplot(1, 3, 3);
	plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
	hold on;
	
	LL4 = [LL4(:)', {"f0 = "+num2str(f/1e9) + " GHz"}];
	
end

figure(3);
subplot(1, 3, 1);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("Fundamental's Frequency Dependence, P="+num2str(P_RF) + " dBm");
grid on;
subplot(1, 3, 2);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("2nd Harmonic's Frequency Dependence, P="+num2str(P_RF) + " dBm");
grid on;
subplot(1, 3, 3);
legend(LL4{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("3rd Harmonic's Frequency Dependence, P="+num2str(P_RF) + " dBm");
grid on;

%% Estimate Conversion Efficiency

figure(4);
subplot(2, 1, 1);
hold off;
subplot(2, 1, 2);
hold off;

figure(5);
subplot(1, 2, 1);
hold off;
subplot(1, 2, 2);
hold off;

% Scan over all powers
CM = resamplecmap(CMAP_CE, numel(powers_dBm));
LL4 = cell(1, numel(powers_dBm));
for pidx = 1:numel(powers_dBm)
	pwr = powers_dBm(pidx);

	% Generate local conditions struct
	c = defaultConditions();
	c.SG_power = pwr;
	c.convert_to_W = 1;
	c.Vnorm = NORMAL_VOLTAGE;
	
% 	multiWaitbar('Generate Figure 4', (pidx-1)/numel(powers_dBm));
	
	CE2 = zeros(1, numel(freqs));
	CE3 = zeros(1, numel(freqs));
	
	% Scan over all frequencies
	idx = 0;
	for f = freqs
		idx = idx + 1;
		
% 		multiWaitbar('Analyze Power level', (idx-1)/numel(freqs));
		
		% Update conditions
		c.f0 = f;
		
		% Extract data
		[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
		
		% Calculate conversion efficiency
		CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
		CE3_ = abs(harm_struct.h3)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
		[CE2(idx), mi2] = max(CE2_);
		[CE3(idx), mi3] = max(CE3_);
		
		% Add to figure 5 (as neccesary)
		if f == FREQ
			figure(5);
			subplot(1, 2, 1);
			plot(Vsweep, CE2_, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
			hold on;
			subplot(1, 2, 2);
			plot(Vsweep, CE3_, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
			hold on;
		end
		
	end
	
	% Plot this power level
	figure(4);
	subplot(2, 1, 1);
	plot(freqs./1e9, CE2, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
	hold on;
	subplot(2, 1, 2);
	plot(freqs./1e9, CE3, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
	hold on;
	
	LL4{pidx} ="P = "+num2str(pwr) + " dBm";
	
end
% multiWaitbar('CloseAll');

figure(4);
subplot(2, 1, 1);
xlabel("Frequency (GHz)");
ylabel("Maximum Conversion Efficiency (%)");
title("2nd Harmonic Conversion Efficiency");
grid on;
legend(LL4{:});
subplot(2, 1, 2);
xlabel("Frequency (GHz)");
ylabel("Maximum Conversion Efficiency (%)");
title("3rd Harmonic Conversion Efficiency");
grid on;
legend(LL4{:});

figure(5);
subplot(1, 2, 1);
xlabel("Bias Voltage (V)");
ylabel("Conversion Efficiency (%)");
title("2nd Harmonic, f = " + num2str(FREQ/1e9) + " GHz");
grid on;
legend(LL4{:});
subplot(1, 2, 2);
xlabel("Bias Voltage (V)");
ylabel("Conversion Efficiency (%)");
title("3rd Harmonic, f = " + num2str(FREQ/1e9) + " GHz");
grid on;
legend(LL4{:});

if CONVERT_V1
	clear ds;
end




















