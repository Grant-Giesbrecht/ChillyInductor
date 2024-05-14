%% SC14
%
% Previously named DS21_VF_versus_bias.m
%
% Looks at VF versus freq and (meas vs AWR vs theory w/ feedback)
%
% Built off of DS20. Looks at how changes in bias impact phase velocity and
% the fundamental tone.
%
% Also plots measured and AWR data - this section borrows heavily from
% SC12.


%% Set Conditions

% Simulation variables
% freqs = linspace(10e9, 11e9, 301);

% VF variables
L0 = 1e-6;
C_ = 121e-12;
q = 0.190;
f0 = 10e9;

% System Variables
chip_len = 0.5;
Z0_chip = 88.4;
Z0_cable = 50;
P_dBm = 8;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);
Zsrc = 50; % Generator source impedance
 
% % Optimization parameters
% Iac_tol = 1e-5; % [A] tolerance for convergence
% conv_coef = 0.5; % Coefficient multiplying error for correction

Ibias_mA = -30:.05:30;%linspace(-30, 30, 61);
Ibias = Ibias_mA.*1e-3;

%% Estimate VF
Vp = 1./sqrt(L0.*C_).*(1-Ibias.^2./q.^2);
VF = Vp./3e8;

figure(1);
hold off;
plot(Ibias.*1e3, VF, 'Color', [0, .6, 0]);
grid on;
xlabel("Bias Current (mA)");
ylabel("Velocity Factor (1)");
title("Velocity Factor versus Frequency");
grid on;
ylim([0.25, 0.35]);

%% Create Microsim Analysis - Current versus Bias, compensating for NOnlinearity

% Define length (degrees)
theta_cable = 45;
theta_chip  = 360.*chip_len./f2l(f0, VF);

% Define elements
load_elmt = shuntRes(50, VF);
cable = tlin(50, theta_cable, VF, true);
cable2 = tlin(50, theta_cable, VF, true);
chip =  tlin(Z0_chip, theta_chip, VF, true);

% Input impedance looking into 50ohm connected to load
net = copyh(load_elmt);
net.series(cable);
zin_A = net.Zin();

% Input ipmedance looking into the chip from generator side
net.series(chip);
zin_B = net.Zin();

% Input ipmedance looking into system
net.series(cable2);
zin_C = net.Zin();

% Calculate expect power and current
P0 = 1./2.*Vgen.^2.*real(zin_C)./( (real(zin_C) + real(Zsrc)).^2 + (imag(zin_C) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
Iac = sqrt(P0./50);

%% Read AWR and Measured Data - Borrows heavily from SC12

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

figure(2);
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
	Ibias_meas = Vdcs.*iv_conv;
	
	% Convert VNA's funky units to real units (2nd harmonic)
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	a_SG = sqrt(cvrt(pwr, 'dBm', 'W'));
	S21 = abs(fund).*a2./a_SG;	
	
	% Calculate things to plot
	P_rec = (S21.*a_SG).^2;
	
	% Create plot
	figure(2);
	plot(Ibias_meas.*1e3, cvrt(P_rec, 'W', 'dBm'), 'Marker', 'o', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	% Add to legend
	legend_list = [legend_list(:)', {strcat("Measured: P = ", num2str(pwr), " dBm")}];

end

unit_0 = 'dBW';
unit_f = 'dBm';

load('C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\6dBm_AWR_Inductor_Model_10GHz_1GHz_comparison.mat');
% plot(I_DC_mA, cvrt(Pfund_1GHz, unit_0, unit_f), 'Marker', '+', 'LineStyle', '--', 'LineWidth', 1.2, 'Color', [0.5, 0, 0]);
plot(I_DC_mA, cvrt(Pfund_10GHz, unit_0, unit_f), 'Marker', 'x', 'LineStyle', '-.', 'LineWidth', 1.4, 'Color', [0.9, 0, 0]);

% figure(1);
% xlabel("Bias Current (mA)");
% ylabel("Power (dBm)");
% title("Comparison of AWR Simulation to Measured Values");
% grid on;
% 
% ylim([-60, 10]);

%% Plot Result

P_expected = Iac.^2.*50;

figure(2);
% plot(Ibias.*1e3, Iac.*1e3, 'Color', [.6, 0, 0]);
plot(Ibias.*1e3, cvrt(P_expected, 'W', 'dBm'), 'LineStyle', '-', 'Color', [0, .5, 0]);
grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Fundamental Power at 10 GHz");
grid on;
legend(legend_list{:}, 'AWR Simulation', strcat('Expected (P = ', num2str(P_dBm), ' dBm)'));
% force0y;
% ylim([0.25, 0.35]);







