%% Description of DS37
% Plots a comparison of measured data and data from an ADS nonlinear
% inductor simulation with 100K stages. Both the measurement and simulation
% use 0 dBm of RF input power.
%

%% System Parameters

USE_LOSS = true;

P_rf_gen_dBm = 0;
Ibias_A = linspace(-30, 30, 61).*1e-3;

P_rf_gen = cvrt(P_rf_gen_dBm, 'dBm', 'W');
Vgen = sqrt(P_rf_gen*200);

l_phys = 0.5;
C_ = 121e-12;
L0 = 900e-9;
q = 0.190;

f1 = 10e9;
ZL = 50;
Zsrc = 50;

Iac_conv = 1e-4;
conv_param1 = 0.5; % 0 = don't change guess, 1 = use new result as guess
conv_param2 = 0.75; % How much to reduce conv_param1 when conversion fails

pi = 3.1415926535;


%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

load(dataset_path("ads_sweep2.mat"));

load(dataset_path("cryostat_sparams.mat"));

%% Unpack measurements

iv_conv = 9.5e-3; % A/V
c = struct('SG_power', -10);
c.Vnorm = 2e-3;
c.SG_power = 0;

% Calculate harmonics over bias sweep
[harms, norm, Vdcs] = getHarmonicSweep(ld, c);
fund = harms.h1;
Ibias = Vdcs.*iv_conv;

%% Apply loss data to ADS results

idx_h1 = findClosest(freq_Hz, f1);
idx_h2 = findClosest(freq_Hz, f1*2);
idx_h3 = findClosest(freq_Hz, f1*3);

S21_h1 = S21_dB(idx_h1);
S21_h2 = S21_dB(idx_h2);
S21_h3 = S21_dB(idx_h3);

displ("Measured Loss:");
displ("  10 GHz: ", S21_h1, " dB");
displ("  20 GHz: ", S21_h2, " dB");
displ("  30 GHz: ", S21_h3, " dB");

if ~USE_LOSS
	S21_h1 = 0;
	S21_h2 = 0;
	S21_h3 = 0;
end

displ("Applying Loss:");
displ("  10 GHz: ", S21_h1, " dB");
displ("  20 GHz: ", S21_h2, " dB");
displ("  30 GHz: ", S21_h3, " dB");

%% Plot all data

lw = 1.5;
mks = 10;

figure(1);
hold off;
plot(Vdcs, VNA2dBm(abs(harms.h1)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
hold on;
plot(Vdcs, VNA2dBm(abs(harms.h2)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
plot(Vdcs, VNA2dBm(abs(harms.h3)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0.7, 0, 0], 'MarkerSize', mks);

plot(Vdc, h1+S21_h1, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0, .5], 'MarkerSize', mks);
hold on;
plot(Vdc, h2+S21_h2, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0.5, 0], 'MarkerSize', mks);
plot(Vdc, h3+S21_h3, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.5, 0, 0], 'MarkerSize', mks);
grid on;
legend("Meas. Fundamental", " Meas. 2nd Harmonic", " Meas. 3rd Harmonic", "Sim. Fundamental", "Sim. 2nd Harmonic", "Sim. 3rd Harmonic");
xlabel("DC Bias Current (mA)");
ylabel("Voltage (V)");
title("Comparison of 0 dBm measurement to 100K ADS Simulation");

% Find all points, exclude -100 (zero power), and calculate minimum point
% for Y-limits.
all_points = [VNA2dBm(abs(harms.h1)), VNA2dBm(abs(harms.h2)), VNA2dBm(abs(harms.h3)), h1+S21_h1, h2+S21_h2, h3+S21_h3];
idxs = (all_points <= -100);
all_points(idxs) = 0;
min_val = min(all_points);
min_lim = floor(min_val/10)*10;
yl0 = ylim();
ylim([min_lim, yl0(2)]);




