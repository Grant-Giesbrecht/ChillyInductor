%% SC10 Predict Currents V2
%
% Previously named: DS18_Predict_Currents_V2.m
%
% Builds off of SC6_Predict_Currents, but fixes the predictions by adding
% in the methodology developed in SC9 which uses the dot product to
% accurately account for phase differences between I and V. It also fixes
% how transmitted power was calcualted. Using eq. 2,76 from Pozar 4e was
% neccesary.

%% Set Conditions

P_dBm = 8.389;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);
Zsrc = 50; % Generator source impedance

plot_input_cable = false;

Z0_chip = 88.4;
Z0_cable = 50;

%% Prepare Sweep Conditions

% N_cable = 90;
% N_chip = 90;
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
cable2 = tlin(50, [conditions.cable], conditions, true);
% cable = tlin(48, 45, conditions, true);
chip =  tlin(Z0_chip, [conditions.chip], conditions, true);
% coupler = seriesRes(90, conditions);

% Input impedance looking into 50ohm connected to load
% net = copyh(cable);
net = copyh(load);
net.series(cable);
% zin_A = net.Zin(50);
zin_A = net.Zin();
displ("Zin into load cable: ", num2fstr(zin_A(1)), " ohms");

% Input ipmedance looking into the chip from generator side
net.series(chip);
% zin_B = net.Zin(50);
zin_B = net.Zin();
displ("Zin into Chip: ", num2fstr(zin_B(1)), " ohms");

% Input ipmedance looking into system
net.series(cable2);
% net.series(coupler)
% zin_C = net.Zin(50);
zin_C = net.Zin();
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

figure(1);

subplot(3, 1, 1);
hold off
plot(cable_vals(I_cable), abs(zin_A(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("Load Cable Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))
force0y;

subplot(3, 1, 2);
hold off
plot(cable_vals(I_cable), abs(zin_B(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("Chip Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))
force0y;

subplot(3, 1, 3);
hold off
plot(cable_vals(I_cable), abs(zin_C(I_cable)));
xlim([0, max(freqs_cable)]);
grid on;
title("System Input Impedance");
ylabel("Impedance (Ohms");
xlabel("Cable length (\lambda)");
set(gca,'Xtick',0:45:freqs_cable(end))
force0y;

%% Calculate Currents Along Chip

% Find resistive equivalent power
% P0 = 0.5*Vgen.^2.*zin_C./(zin_C + Zsrc).^2; % Total power (multiplied by half because I and V were not RMS)
P0 = 1./2.*Vgen.^2.*real(zin_C)./( (real(zin_C) + real(Zsrc)).^2 + (imag(zin_C) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
gamma_1D = Z2G(zin_C, Zsrc);

N_points = 3;

% Find Input Impedances along chip (For Finding Current)
ZL = 50; % Starting load impedance - 50 ohms looking towards load
Zchip_0 = xfmr2zin(Z0_chip, ZL, pi./180.*[conditions.chip]./N_points);
Zchip_1 = xfmr2zin(Z0_chip, ZL, pi./180.*[conditions.chip]./N_points.*2);
Zchip_2 = xfmr2zin(Z0_chip, ZL, pi./180.*[conditions.chip]./N_points.*3);

% Find Input Impedances along Input Cable (For Finding Current)
ZL = Zchip_2; % Starting load impedance - 50 ohms looking towards load
Zcable_0 = xfmr2zin(Z0_cable, ZL, pi./180.*[conditions.cable]./N_points);
Zcable_1 = xfmr2zin(Z0_cable, ZL, pi./180.*[conditions.cable]./N_points.*2);
Zcable_2 = xfmr2zin(Z0_cable, ZL, pi./180.*[conditions.cable]./N_points.*3);

% Now find Vx and Ix at each point

% All impedance values (including load)
Z0s = {Zchip_0, Zchip_1, Zchip_2};
if plot_input_cable
	Z0s = {Zcable_0, Zcable_1, Zcable_2};
end

% Prepare output data
Ixs = {};
Vxs = {};

% For each stage
idx = 0;
for Zx_c = Z0s
	idx = idx + 1;
	
	% Convert cell to array
	Zx = Zx_c{:};
	
	% Find angle - magnitude will be wrong!
	Ix = sqrt(2.*P0./Zx); % RMS Value
	Vx = sqrt(2.*P0.*Zx); % RMS Value
	
	% Find angle between vectors
	theta = angle(Ix./Vx);
	
	% Find Vx*Ix (NOT dot product)
	Px = P0./cos(theta);
	
	% Recalculate ix and Vx with correct amplitude
	Ixs(idx) = {sqrt(2.*Px./Zx)}; % RMS Value
	Vxs(idx) = {sqrt(2.*Px.*Zx)}; % RMS Value
end

%% Convert 1D vectors to 2D vectors for plotting

% Unpack V and I data
Vx0_1D = Vxs(1);
Vx0_1D = Vx0_1D{:};
Ix0_1D = Ixs(1);
Ix0_1D = Ix0_1D{:};

Vx1_1D = Vxs(2);
Vx1_1D = Vx1_1D{:};
Ix1_1D = Ixs(2);
Ix1_1D = Ix1_1D{:};

Vx2_1D = Vxs(3);
Vx2_1D = Vx2_1D{:};
Ix2_1D = Ixs(3);
Ix2_1D = Ix2_1D{:};

% Generate mesh
[X, Y] = meshgrid(freqs_cable, freqs_chip);

FI_chip = find(I_chip);

% Generate solution set
Z = zeros(N_chip, N_cable);
Psys = zeros(N_chip, N_cable);
gamma = zeros(N_chip, N_cable);
Vx0 = zeros(N_chip, N_cable);
Ix0 = zeros(N_chip, N_cable);
Vx1 = zeros(N_chip, N_cable);
Ix1 = zeros(N_chip, N_cable);
Vx2 = zeros(N_chip, N_cable);
Ix2 = zeros(N_chip, N_cable);
for c = 1:N_cable % For each column...
	start_idx = (c-1)*N_chip+1;
	Z(:, c) = zin_C(start_idx:start_idx+N_chip-1); % Save solution data
	
	Psys(:, c) = P0(start_idx:start_idx+N_chip-1); % Save solution data
	gamma(:, c) = gamma_1D(start_idx:start_idx+N_chip-1); % Save solution data
	
	Vx0(:, c) = Vx0_1D(start_idx:start_idx+N_chip-1); % Save solution data
	Ix0(:, c) = Ix0_1D(start_idx:start_idx+N_chip-1); % Save solution data
	
	Vx1(:, c) = Vx1_1D(start_idx:start_idx+N_chip-1); % Save solution data
	Ix1(:, c) = Ix1_1D(start_idx:start_idx+N_chip-1); % Save solution data
	
	Vx2(:, c) = Vx2_1D(start_idx:start_idx+N_chip-1); % Save solution data
	Ix2(:, c) = Ix2_1D(start_idx:start_idx+N_chip-1); % Save solution data
end

%% Generate 3-D Plot - System Input

figure(5);
colormap parula;
surf(X,Y,abs(Psys.*1e3));
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("Power (mW)");
title("Power Delivered to System");
set(gca,'Xtick',0:45:freqs_cable(end))
set(gca,'Ytick',0:45:freqs_chip(end))
shading flat;

figure(6);
colormap parula;
surf(X,Y,abs(gamma));
xlabel("Cable Length (deg)");
ylabel("Chip Length (deg)");
zlabel("|\Gamma|");
title("Magnitude of System Reflection");
set(gca,'Xtick',0:45:freqs_cable(end))
set(gca,'Ytick',0:45:freqs_chip(end))
shading flat;


%% Generate 3-D Plot - V & I Inside Chip

if ~plot_input_cable
	CM_current = autumn;
	CM_voltage = spring;

	figure(2);
	ax2_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx0));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("1/3 Into Chip From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax2_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix0.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("1/3 Into Chip From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	figure(3);
	ax3_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx1));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("2/3 Into Chip From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax3_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix1.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("2/3 Into Chip From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	figure(4);
	ax4_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx2));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("Input to Chip");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax4_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix2.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("Input to Chip");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	colormap(ax2_1, CM_voltage);
	colormap(ax2_2, CM_current);
	colormap(ax3_1, CM_voltage);
	colormap(ax3_2, CM_current);
	colormap(ax4_1, CM_voltage);
	colormap(ax4_2, CM_current);
end

%% Generate 3-D Plot - V & I Inside Input Cable

if plot_input_cable
	CM_current = autumn;
	CM_voltage = summer;

	figure(2);
	ax2_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx0));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("1/3 Into Input Cable From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax2_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix0.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("1/3 Into Input Cable From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	figure(3);
	ax3_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx1));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("2/3 Into Input Cable From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax3_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix1.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("2/3 Into Input Cable From Load");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	figure(4);
	ax4_1 = subplot(1, 2, 1);
	surf(X,Y,abs(Vx2));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Voltage Amplitude (V)");
	title("Input to System");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	ax4_2 = subplot(1, 2, 2);
	surf(X,Y,abs(Ix2.*1e3));
	xlabel("Cable Length (deg)");
	ylabel("Chip Length (deg)");
	zlabel("AC Current Amplitude (mA)");
	title("Input to System");
	set(gca,'Xtick',0:45:freqs_cable(end))
	set(gca,'Ytick',0:45:freqs_chip(end))
	shading flat;

	colormap(ax2_1, CM_voltage);
	colormap(ax2_2, CM_current);
	colormap(ax3_1, CM_voltage);
	colormap(ax3_2, CM_current);
	colormap(ax4_1, CM_voltage);
	colormap(ax4_2, CM_current);
end





















