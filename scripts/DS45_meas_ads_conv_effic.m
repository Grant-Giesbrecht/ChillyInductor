%% Built off of DS37 to estimate conversion efficiency

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

%% Apply loss to find power in and out at chip plane

% measurements

pchip_m1 = VNA2dBm(abs(harms.h1)) - S21_h1/2;
pchip_m2 = VNA2dBm(abs(harms.h2)) - S21_h2/2;
pchip_m3 = VNA2dBm(abs(harms.h3)) - S21_h3/2;

pchip_s1 = h1+S21_h1 - S21_h1/2;
pchip_s2 = h2+S21_h2 - S21_h2/2;
pchip_s3 = h3+S21_h3 - S21_h3/2;

pin_m1 = VNA2dBm(abs(harms.h1)) - S21_h1;
pin_m2 = VNA2dBm(abs(harms.h2)) - S21_h2;
pin_m3 = VNA2dBm(abs(harms.h3)) - S21_h3;

pin_s1 = h1+S21_h1 - S21_h1;
pin_s2 = h2+S21_h2 - S21_h2;
pin_s3 = h3+S21_h3 - S21_h3;

% pchip_mCE = dB2lin(pchip_m2, 10)./(dB2lin(pchip_m1,10) + dB2lin(pchip_m2,10) + dB2lin(pchip_m3,10) ).*100;
% pvna_mCE = dB2lin(VNA2dBm(abs(harms.h2)), 10)./(dB2lin(VNA2dBm(abs(harms.h1)),10) + dB2lin(VNA2dBm(abs(harms.h2)),10) + dB2lin(VNA2dBm(abs(harms.h3)), 10)).*100;
% pchip_sCE = dB2lin(h2, 10)./(dB2lin(h1, 10)+dB2lin(h2, 10)+dB2lin(h3, 10)).*100;

pchip_mCE = cvrt(pchip_m2, 'dBm', 'W')./(cvrt(pchip_m1, 'dBm', 'W') + cvrt(pchip_m2, 'dBm', 'W') + cvrt(pchip_m3, 'dBm', 'W') ).*100;
pvna_mCE = cvrt(VNA2dBm(abs(harms.h2)), 'dBm', 'W')./(cvrt(VNA2dBm(abs(harms.h1)), 'dBm', 'W') + cvrt(VNA2dBm(abs(harms.h2)), 'dBm', 'W') + cvrt(VNA2dBm(abs(harms.h3)), 'dBm', 'W')).*100;
pchip_sCE = cvrt(h2, 'dBm', 'W')./(cvrt(h1, 'dBm', 'W')+cvrt(h2, 'dBm', 'W')+cvrt(h3, 'dBm', 'W')).*100;
pchip_mCE_fixRFin = cvrt(pchip_m2, 'dBm', 'W')./cvrt(0, 'dBm', 'W').*100;
pchip_mCE_chip_over_in = cvrt(pchip_m2, 'dBm', 'W')./(cvrt(pin_m1, 'dBm', 'W') + cvrt(pin_m2, 'dBm', 'W') + cvrt(pin_m3, 'dBm', 'W')).*100;

pchip_mCE3 = cvrt(pchip_m3, 'dBm', 'W')./(cvrt(pchip_m1, 'dBm', 'W') + cvrt(pchip_m2, 'dBm', 'W') + cvrt(pchip_m3, 'dBm', 'W') ).*100;
pchip_sCE3 = cvrt(h3, 'dBm', 'W')./(cvrt(h1, 'dBm', 'W')+cvrt(h2, 'dBm', 'W')+cvrt(h3, 'dBm', 'W')).*100;

%% Plot Standard data

lw = 1.5;
mks = 10;

c_conv = [0.4, 0, 0.5];
mk_conv = '.';
mkz_conv = 30;

figure(1);
subplot(1, 2, 1);
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
ylabel("Power (dBm)");
title("VNA Plane Powers");
ylim([-75, 0]);

subplot(1, 2, 2);
hold off;
plot(Vdcs, pvna_mCE, 'LineStyle', '--', 'LineWidth', lw, 'Marker', mk_conv, 'Color', c_conv, 'MarkerSize', mkz_conv);
grid on;
xlabel("DC Bias Current (mA)");
ylabel("Conversion Efficiency (%)");
title("VNA Plane Powers");




%% Plot Chip-Plane data

lw = 1.5;
mks = 10;

figure(2);
subplot(1, 2, 1);
hold off;
plot(Vdcs, pchip_m1, 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
hold on;
plot(Vdcs, pchip_m2, 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
plot(Vdcs, pchip_m3, 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0.7, 0, 0], 'MarkerSize', mks);

plot(Vdc, pchip_s1, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0, .5], 'MarkerSize', mks);
hold on;
plot(Vdc, pchip_s2, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0.5, 0], 'MarkerSize', mks);
plot(Vdc, pchip_s3, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.5, 0, 0], 'MarkerSize', mks);
grid on;
legend("Meas. Fundamental", " Meas. 2nd Harmonic", " Meas. 3rd Harmonic", "Sim. Fundamental", "Sim. 2nd Harmonic", "Sim. 3rd Harmonic");
xlabel("DC Bias Current (mA)");
ylabel("Power (dBm)");
title("Chip Plane Harmonic Power (P_{RF} = 0 dBm)");
ylim([-75, 0]);

subplot(1, 2, 2);
hold off
plot(Vdcs, pchip_mCE, 'LineStyle', '--', 'LineWidth', lw, 'Marker', mk_conv, 'Color', c_conv, 'MarkerSize', mkz_conv);
hold on;
plot(Vdc, pchip_sCE, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', c_conv, 'MarkerSize', 10);
plot(Vdcs, pchip_mCE_fixRFin, 'LineStyle', '--', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.6, 0, 0], 'MarkerSize', 8);
plot(Vdcs, pchip_mCE_chip_over_in, 'LineStyle', '--', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0, 0.6], 'MarkerSize', 8);
grid on;
xlabel("DC Bias Current (mA)");
ylabel("Conversion Efficiency (%)");
title("Chip Plane Measured Conversion Efficiency");
legend("Measurement", "Simulated", "measured vs SG-power", "measured vs Input Plane");

%% Linear Scale

figure(3);
hold off;
yyaxis left;
plot(Vdcs, cvrt(pchip_m1, 'dBm', 'mW'), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
ylabel("1st Harmonic Power (mW)");
yyaxis right;
hold off;
plot(Vdcs, cvrt(pchip_m2, 'dBm', 'mW'), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
hold on;
plot(Vdcs, cvrt(pchip_m3, 'dBm', 'mW'), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0.7, 0, 0], 'MarkerSize', mks);
legend("Meas. Fundamental", " Meas. 2nd Harmonic", " Meas. 3rd Harmonic");
xlabel("DC Bias Current (mA)");
ylabel("2nd & 3rd Harmonic Power (mW)");

title("Chip Plane Harmonic Power (P_{RF} = 0 dBm) - Linear Scale");

yyaxis left;
ax = gca;
ax.YColor = 'k';
grid on;
ylim([0, 0.5]);
yyaxis right;
ax = gca;
ax.YColor = 'k';
ylim([0, 0.05]);
yyaxis left;

%% 3rd harmonic conversion efficiency

c_conv3 = [0, 0.6, 0];
mkz_conv = 20;
figure(4);
hold off
plot(Vdcs.*iv_conv.*1e3, pchip_mCE, 'LineStyle', '--', 'LineWidth', lw, 'Marker', mk_conv, 'Color', c_conv, 'MarkerSize', mkz_conv);
hold on;
plot(Vdc.*iv_conv.*1e3, pchip_sCE, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', c_conv, 'MarkerSize', 10);
plot(Vdcs.*iv_conv.*1e3, pchip_mCE3, 'LineStyle', '--', 'LineWidth', lw, 'Marker', mk_conv, 'Color', c_conv3, 'MarkerSize', mkz_conv);
plot(Vdc.*iv_conv.*1e3, pchip_sCE3, 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', c_conv3, 'MarkerSize', 10);
% plot(Vdcs, pchip_mCE_fixRFin, 'LineStyle', '--', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.6, 0, 0], 'MarkerSize', 8);
% plot(Vdcs, pchip_mCE_chip_over_in, 'LineStyle', '--', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0, 0.6], 'MarkerSize', 8);
grid on;
xlabel("DC Bias Current (mA)");
ylabel("Conversion Efficiency (%)");
title("2nd and 3rd Harmonic Conversion Efficiency, P_{RF} = 0 dBm");
legend("Measurement, 2f_0", "Simulated, 2f_0", "Measurement, 3f_0", "Simulation, 3f_0");





































