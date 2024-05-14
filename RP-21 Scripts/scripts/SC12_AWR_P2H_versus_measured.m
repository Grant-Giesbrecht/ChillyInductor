% Previously DS19_AWR_P2H_versus_measured
%
% Plots data from AWR predicting power at the 2nd harmonic from AWR's
% nonlinar sim versus the measured data in DS5.

% Import data
load(dataset_path("DS5_FinePower_PO-1.mat"));

% Get power levels
pwr_all = unique([ld.SG_power_dBm]); % [dBm]
pwr_plot = [4, 6, 8];

% Nathan's assumed parameters
len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

f = 10e9;

% Create conditoins
c = struct('SG_power', -10);
c.Vnorm = 2e-3;

pwr_VNA_dBm = -10; %[dBm]

figure(1);
hold off;

legend_list = {};

% Loop over each power level
CM = resamplecmap(colormap('cool'), numel(pwr_plot));
idx = 0;

for pwr = pwr_plot
	idx = idx + 1;
	
	% Filter Data - Generate Plot: Power vs DC bias

	% Set SG power parameter
	c.SG_power = pwr;
	
	% Calculate harmonics over bias sweep
	[harms, norm, Vdcs] = getHarmonicSweep(ld, c);
	fund = harms.h1;
	h2 = harms.h2;
	Ibias = Vdcs.*iv_conv;
	
	% Convert VNA's funky units to real units (2nd harmonic)
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	a_SG = sqrt(cvrt(pwr, 'dBm', 'W'));
	S21 = abs(h2).*a2./a_SG;	
	
	% Calculate things to plot
	P_rec = (S21.*a_SG).^2;
	
	% Create plot
	figure(1);
	plot(Ibias.*1e3, cvrt(P_rec, 'W', 'dBm'), 'Marker', 'o', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	% Add to legend
	legend_list = [legend_list(:)', {strcat("Measured: P = ", num2str(pwr), " dBm")}];

end

unit_0 = 'dBW';
unit_f = 'dBm';

load('C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\6dBm_AWR_Inductor_Model_10GHz_1GHz_comparison.mat');

plot(I_DC_mA, cvrt(Pfund_1GHz, unit_0, unit_f), 'Marker', '+', 'LineStyle', '--', 'LineWidth', 1.2, 'Color', [0.5, 0, 0]);
plot(I_DC_mA, cvrt(Pfund_10GHz, unit_0, unit_f), 'Marker', 'x', 'LineStyle', '-.', 'LineWidth', 1.4, 'Color', [0.9, 0, 0]);
plot(I_DC_mA, cvrt(P2H_1GHz, unit_0, unit_f), 'Marker', '+', 'LineStyle', '--', 'LineWidth', 1.2, 'Color', [0, 0, 0.5]);
plot(I_DC_mA, cvrt(P2H_10GHz, unit_0, unit_f), 'Marker', 'x', 'LineStyle', '-.', 'LineWidth', 1.4, 'Color', [0, 0, 0.9]);

figure(1);
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Comparison of AWR Simulation to Measured Values");
grid on;
legend(legend_list{:}, 'AWR Sim: 1 GHz Fundamental', 'AWR Sim: 10 GHz Fundamental', 'AWR Sim: 1 GHz 2nd Harmoinc', 'AWR Sim: 10 GHz 2nd Harmonic');
ylim([-60, 10]);







