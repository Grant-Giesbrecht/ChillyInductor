%% Simulate the currents in a series of impedance transformers
%
% Look at the total current in (based on resistive model), and current in
% each stage (considering Z0 of each stage).
%
% Assumes lossless, so uses principle that Vx=Ix*Z0 and Vx*Ix=P s.t. P is the
% total (constant) power passing through the system, and Vx and Ix are the
% voltage and current in a given stage w/ impedance Z0.

%% Find input impedance for system
%
% System:
%   [50 ohm port] -- [30 ohm, 90 deg TLIN] -- [10 ohm, 90 deg TLIN] -- [ 50 ohm load]
%
% Note this V and I that will be calculated are not 'physical'. They
% represent the power, and are the V and I should it be a resistor, but in
% the physical system with weird TLIN transformers, they'll be different!

% Calculate input impedances
Z1 = xfmr2zin(10, 50, pi/2); % Transform first stage
Z2 = xfmr2zin(30, Z1, pi/2); % Transform second stage
displ("Input Impedance to 10 ohm line: ", num2fstr(Z1), " ohms");
displ("Input Impedance to 30 ohm line: ", num2fstr(Z2), " ohms");

Zin = Z2;
Vg = 1; % Pretend 1 V source, doesn't matter
Zsrc = 50; % Generator impedance

% Calculate resistive-equivalent V&I
I_res = Vg./(Zin + Zsrc);
V_res = Zin./(Zin + Zsrc);
P0 = Vg.*Zin./(Zin + Zsrc).^2; % Total power

displ("Resistive Equivalents:");
displ("  Ir = ", num2fstr(I_res.*1e3), " mA")
displ("  Vr = ", num2fstr(V_res), " V")

%% Calculate Physical V&I in each stage

% All impedance values (including load)
Z0s = [30, 10, 50];

% For each stage
for Z0 = Z0s
	
	Ix = sqrt(Vg.*Zin./Z0)./(Zin + Zsrc);
	Vx = sqrt(Vg.*Zin.*Z0)./(Zin + Zsrc);
	
	displ("Z0 = ", Z0, ":");
	displ("  Ix = ", num2fstr(Ix.*1e3), " mA")
	displ("  Vx = ", num2fstr(Vx), " V")
	
end



















