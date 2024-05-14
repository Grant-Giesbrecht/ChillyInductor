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
load = shuntRes(50, conditions);
cable = tlin(50, [conditions.cable], conditions, true);
chip =  tlin(90, [conditions.chip], conditions, true);

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
net.series(cable);
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