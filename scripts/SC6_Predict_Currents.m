%% SC6 Predict Currents
%
% Previously named DS11_predict_currents_experiment.m
%
% Models the kinetic inductance system as a series of transmission lines
% and predicts the input impedance and current at VNA that results. 

%% Predict Circuit Parameters (Will compare to Microsim solution)

P_dBm = 8.389;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);

Zin = 36 - 11.4i;

VL = Zin/(Zin + 50)*Vgen;
IL = Vgen/(Zin + 50);

barprint(strcat("P = ", num2fstr(P_dBm), " dBm (", num2fstr(P_watts*1e3), " mW)"));
displ("Results for expected cable length:");
displ();
displ("At system input:");
displ("  |V| = ", abs(VL), " V");
displ("  |I| = ", abs(IL*1e3), " mA");
displ();
displ("Along Chip:");
displ("  |V| = ", abs(VL*1.4), " V");
displ("  |I| = ", abs(IL/1.4*1e3), " mA");

%% Prepare Sweep Conditions

N_cable = 360;
N_chip = 360;

freqs_cable = linspace(1, 360, N_cable);
freqs_chip = linspace(1, 360, N_chip);

conditions = struct('cable', NaN, 'chip', NaN);
count = 1;
for i = freqs_cable
	for k = freqs_chip
		conditions(count) = struct('cable', i, 'chip', k);
		
		count = count + 1;
	end
end

%% Run Microsim Analysis

% Define length
theta_cable = freqs_cable;
theta_chip  = freqs_cable;

% Define elements
load = shuntRes(51, conditions);
cable = tlin(48, [conditions.cable], conditions, true);
cable2 = tlin(48, [conditions.cable], conditions, true);
chip =  tlin(88.4, [conditions.chip], conditions, true);
coupler = seriesRes(90, conditions);

% Input impedance looking into 50ohm connected to load
net = copyh(cable);
% net = copyh(load);
% net.series(cable);
zin_A = net.Zin(50);
displ("Zin into load cable: ", num2fstr(zin_A(1)), " ohms");

% Input ipmedance looking into the chip from generator side
net.series(chip);
zin_B = net.Zin(50);
displ("Zin into Chip: ", num2fstr(zin_B(1)), " ohms");

% Input ipmedance looking into system
net.series(cable2);
net.series(coupler)
zin_C = net.Zin(50);
displ("Zin into system: ", num2fstr(zin_C(1)), " ohms");

%% Prepare Masks for Analysis

chip_vals = [conditions.chip];
cable_vals = [conditions.cable];

% Get sweep indecies
angle_chip = 45; % Desired angle
exact_chip = freqs_chip(findClosest(freqs_chip, angle_chip)); % Closest exact anlge
I_cable = chip_vals == exact_chip;

angle_cable = 180; % Desired angle
exact_cable = freqs_cable(findClosest(freqs_cable, angle_cable)); % Closest exact anlge
I_chip = cable_vals == exact_cable;

%% Generate Plots

figure(2);

subplot(3, 1, 1);
hold off
plot(cable_vals(I_cable), abs(zin_A(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("Load Cable Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))

% subplot(3, 1, 2);
% hold off
% plot(chip_vals(I_chip), abs(zin_B(I_chip)));
% xlim([0, max(freqs_cable)]);
% grid on;
% title("Chip Input Impedance");
% ylabel("Impedance (Ohms");
% xlabel("Cable length (\lambda)");
% set(gca,'Xtick',0:45:freqs_cable(end))
% 
% subplot(3, 1, 3);
% hold off
% plot(chip_vals(I_chip), abs(zin_C(I_chip)));
% xlim([0, max(freqs_cable)]);
% grid on;
% title("System Input Impedance");
% ylabel("Impedance (Ohms");
% xlabel("Cable length (\lambda)");
% set(gca,'Xtick',0:45:freqs_cable(end))

subplot(3, 1, 2);
hold off
plot(cable_vals(I_cable), abs(zin_B(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("Chip Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))

subplot(3, 1, 3);
hold off
plot(cable_vals(I_cable), abs(zin_C(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("System Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))

%% Generate 3-D Plot

% Generate mesh
[X, Y] = meshgrid(freqs_cable, freqs_chip);

FI_chip = find(I_chip);

% Generate solution set
Z = zeros(N_chip, N_cable);
for c = 1:N_cable % For each column...
	start_idx = (c-1)*N_chip+1;
	Z(:, c) = zin_C(start_idx:start_idx+N_chip-1); % Save solution data
end

figure(3);
surf(X,Y,abs(Z));
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("System Input Impedance (Ohms)");
set(gca,'Xtick',0:45:freqs_cable(end))
set(gca,'Ytick',0:45:freqs_chip(end))
shading flat;

figure(4);
surf(X,Y,real(Z));
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("System Input Resistance (Ohms)");
set(gca,'Xtick',0:45:freqs_cable(end))
set(gca,'Ytick',0:45:freqs_chip(end))
shading flat;

figure(5);
surf(X,Y,imag(Z));
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("System Input Impedance, Reactance (Ohms)");
set(gca,'Xtick',0:45:freqs_cable(end))
set(gca,'Ytick',0:45:freqs_chip(end))
shading flat;

%% Current surface plot

P_dBm = 6;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);

VL_surf = Z./(Z + 50).*Vgen;
IL_surf = Vgen./(Z + 50);

% Transform currents to account for impedance change
IL_surf_load = IL_surf./50.*Z;

contour_level = 3.9;
% contour_color = [0.6, 0, 0.7];
contour_color = [.85, 0, 0];

figure(6);
hold off;
surfc(X,Y,abs(IL_surf_load.*1e3));
hold on;
contour3(X,Y,abs(IL_surf_load.*1e3), [contour_level, contour_level], 'LineWidth', 2, 'Color', contour_color);
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("VNA Current Estimate (mA)");
set(gca,'Xtick',0:45:freqs_cable(end));
set(gca,'Ytick',0:45:freqs_chip(end));
shading flat;
title("Current Estimate at VNA Input");

figure(7);
hold off;
surfc(X,Y,abs(IL_surf.*1e3));
hold on;
contour3(X,Y,abs(IL_surf.*1e3), [contour_level, contour_level], 'LineWidth', 2, 'Color', contour_color);
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("Generator Current Estimate (mA)");
set(gca,'Xtick',0:45:freqs_cable(end));
set(gca,'Ytick',0:45:freqs_chip(end));
shading flat;
title("Current Estimate at Generator Output");

displ("Min current: ", min(min(abs(IL_surf_load*1e3))), " mA");