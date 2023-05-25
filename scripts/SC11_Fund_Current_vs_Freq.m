%% SC11
%
% Previously named: DS19_Fund_Current_vs_Freq
%
% Now that SC10 was shown that the 50 ohm cables really do have no impact
% on current levels, this is a simplified script to show how (for a given
% chip length), current at the fundamental should change over frequency.
%
% In addition to showing expected currents, it shows them plotted versus
% the measured values (with reasonable agreement, considering the
% imperfectly known length, Z0, etc).
%
% NOTE: This does NOT consider the effect that a change in the fundamental
% current has on the phase velocity, and that's impact on how the chip
% 'looks', and thus its impact on current. See DS20 for that analysis.

%% Set Conditions

P_dBm = 6;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);
Zsrc = 50; % Generator source impedance

chip_len = 0.5;
VF = 0.3;
freqs = linspace(10e9, 11e9, 301);

plot_measured_data = true;

Z0_chip = 88.4;
Z0_cable = 50;

%% Create Microsim Analysis

% Define length (degrees)
theta_cable = 45;
theta_chip  = 360.*chip_len./f2l(freqs, VF);

% Define elements
load_elmt = shuntRes(50, freqs);
cable = tlin(50, theta_cable, freqs, true);
cable2 = tlin(50, theta_cable, freqs, true);
chip =  tlin(Z0_chip, theta_chip, freqs, true);

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

%% Find Currents, Voltages, Input Impedances, Powers, etc

% Find input power
P0 = 1./2.*Vgen.^2.*real(zin_C)./( (real(zin_C) + real(Zsrc)).^2 + (imag(zin_C) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
Iac_exp = sqrt(P0./50);

% Plot P0 over frequency (if system is lossless, this should be power at
% VNA)
figure(1);
hold off;
subplot(1, 1, 1);
plot(freqs./1e9, cvrt(P0, 'W', 'dBm'), 'Color', [0, 0, .6]);
force0y;
xlabel("Frequency (GHz)");
ylabel("Transmitted Power (dBm)");
grid on;
title("Expected Power at VNA");

figure(2);
hold off;
subplot(1, 1, 1);
plot(freqs./1e9, Iac_exp.*1e3, 'Color', [0, 0, .6]);
force0y;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)");
grid on;
title("Expected Current at VNA");

%% Standing Waves Along Line

%% Plot Measured Data

% Exit if don't plot measured data
if ~plot_measured_data
	return;
end

% Save variables from above
freqs_exp = freqs;

% Import data
% load(dataset_path("DS6_10GHzFreqSweep_FF1.mat"));
load(dataset_path("DS7_3BandFreqSweep_FF1.mat"));

SG_pwr = ld(1).SG_power_dBm;

% Create data arrays
all_freq = [ld.SG_freq_Hz];
S21 = zeros(1, numel(ld));

% Get conversion parameters
a2 = sqrt(cvrt(-10, 'dBm', 'W'));
a_SG = sqrt(cvrt(SG_pwr, 'dBm', 'W'));

% Get unique frequencies
freqs = unique(all_freq);
avg_S21 = zeros(1, numel(freqs));
std_S21 = zeros(1, numel(freqs));

% Get mean  of VNA data for b1a2
for idx = 1:numel(S21)
	
	% Get mean of data
	b1a2 = mean(ld(idx).VNA_data.data(1,:));

	% Convert a1b2 to S21
	S21(idx) = abs(b1a2).*a2./a_SG;
	
end

% Average by frequency
idx = 0;
for f = freqs
	idx = idx + 1;
	
	% Get mask
	I = (f == all_freq);
	
	avg_S21(idx) = mean(S21(I));
	std_S21(idx) = std(S21(I));
	
end

% Get V and I
V = avg_S21.*a_SG.*sqrt(50);
Iac = V./50.*sqrt(2);
% Iac = avg_S21./sqrt(50);

figure(3);
hold off;
plot(freqs./1e9, Iac.*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
hold on;
plot(freqs_exp./1e9, Iac_exp.*1e3);
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Measurement vs Expectation Comparison")
force0y;
legend("Measured Current", "Expected Current");
legend("Position", [0.60952,0.39524,0.26429,0.0823])













