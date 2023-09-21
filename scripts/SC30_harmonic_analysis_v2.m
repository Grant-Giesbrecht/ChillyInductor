%% SC30 Harmonic Analysis V2
%
% Built off of SC25 and SC26. Performs data analysis on v2 data,
% calculating conversion efficiency, etc.

%% Configuration Data

P_RF = 2; % dBm
FREQ = 10e9; % Hz

NORMAL_VOLTAGE = 0.00035;

SKIP_READ = true;

CMAP_PWR = 'parula';
CMAP_FREQ = 'jet';
CMAP_CE = 'autumn';

%% Data Location

if ~SKIP_READ || ~exist('ds', 'var')
	DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
	DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');

	filename = 'gamma_14Sept2023_1600.mat';

	load(fullfile(DATA_PATH2, filename));
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

LL = {};
CM = resamplecmap(CMAP_PWR, numel(powers_dBm));
idx = 0;
for pwr = powers_dBm
	idx = idx + 1;
	
	% Update conditions
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
	
	LL = [LL(:)', {"P = "+num2str(pwr) + " dBm"}];
	
end

figure(2);
subplot(1, 3, 1);
legend(LL{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("Fundamental's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
grid on;
subplot(1, 3, 2);
legend(LL{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("2nd Harmonic's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
grid on;
subplot(1, 3, 3);
legend(LL{:});
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

LL = {};
CM = resamplecmap(CMAP_FREQ, numel(freqs));
idx = 0;
for f = freqs
	idx = idx + 1;
	
	% Update conditions
	c.f0 = f;
	
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
	
	LL = [LL(:)', {"f0 = "+num2str(f/1e9) + " GHz"}];
	
end

figure(3);
subplot(1, 3, 1);
legend(LL{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("Fundamental's Frequency Dependence, P="+num2str(P_RF) + " dBm");
grid on;
subplot(1, 3, 2);
legend(LL{:});
xlabel("Bias Voltage (V)");
ylabel("Power at Chip Ouptut (dBm)");
title("2nd Harmonic's Frequency Dependence, P="+num2str(P_RF) + " dBm");
grid on;
subplot(1, 3, 3);
legend(LL{:});
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

% Scan over all powers
CM = resamplecmap(CMAP_CE, numel(powers_dBm));
pidx = 0;
LL = cell(1, numel(powers_dBm));
parfor pidx = 1:numel(powers_dBm)
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
		[CE2(idx), mi2] = max(abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100);
		[CE3(idx), mi3] = max(abs(harm_struct.h3)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100);
		
	end
	
	% Plot this power level
	figure(4);
	subplot(2, 1, 1);
	plot(freqs, CE2, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
	hold on;
	subplot(2, 1, 2);
	plot(freqs, CE3, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :));
	hold on;
	
	LL{pidx} ="P = "+num2str(pwr) + " dBm";
	
end
% multiWaitbar('CloseAll');


























