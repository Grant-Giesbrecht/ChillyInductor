%% Description of 

%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

load(dataset_path("ads_sweep4_phase.mat"));

load(dataset_path("cryostat_sparams.mat"));

%% Unpack measurements

iv_conv = 9.5e-3; % A/V
c = struct('SG_power', -10);
c.Vnorm = 2e-3;
c.SG_power = 4;

% Calculate harmonics over bias sweep
[harms, norm, Vdcs] = getHarmonicSweep(ld, c);

theta = angle(harms.h1).*180./pi;

%% Plot all data

lw = 1.5;
mks = 10;

figure(1);
hold off;
plot(Vdcs, theta-theta(floor(numel(theta)/2)+1), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
hold on;
plot(Vdc, phase_deg-phase_deg(1), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
grid on;
legend("Measured", "ADS Simulation");
xlabel("DC Bias Voltage (V)");
ylabel("Phase (^\circ)");
title("ADS vs Measurement Phase Comparison, P_{RF} = 4 dBm");


