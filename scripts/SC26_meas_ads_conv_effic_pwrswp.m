%% SC26
%
% PReviously named DS46
%
% Shows how conversion efficiency depends on input power.
%
% Built off of DS37 to estimate conversion efficiency

%% System Parameters

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

% Power levels at which to plot CE
power_levels = flip([-6, -4, -2, 0, 2, 4, 6]);
power_levels = flip([-6, -2, 2, 4, 6]);

%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

load(dataset_path("ads_sweep2.mat"));

load(dataset_path("cryostat_sparams.mat"));

%% Unpack measurements

CE_data = {};

idx = 0;
for pwr = power_levels
	idx = idx + 1;
	
	iv_conv = 9.5e-3; % A/V
	c = struct('SG_power', -10);
	c.Vnorm = 2e-3;
	c.SG_power = pwr;

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
	
	pchip_mCE = cvrt(pchip_m2, 'dBm', 'W')./(cvrt(pchip_m1, 'dBm', 'W') + cvrt(pchip_m2, 'dBm', 'W') + cvrt(pchip_m3, 'dBm', 'W') ).*100;
	pvna_mCE = cvrt(VNA2dBm(abs(harms.h2)), 'dBm', 'W')./(cvrt(VNA2dBm(abs(harms.h1)), 'dBm', 'W') + cvrt(VNA2dBm(abs(harms.h2)), 'dBm', 'W') + cvrt(VNA2dBm(abs(harms.h3)), 'dBm', 'W')).*100;
	pchip_sCE = cvrt(h2, 'dBm', 'W')./(cvrt(h1, 'dBm', 'W')+cvrt(h2, 'dBm', 'W')+cvrt(h3, 'dBm', 'W')).*100;
	pchip_mCE_fixRFin = cvrt(pchip_m2, 'dBm', 'W')./cvrt(0, 'dBm', 'W').*100;
	pchip_mCE_chip_over_in = cvrt(pchip_m2, 'dBm', 'W')./(cvrt(pin_m1, 'dBm', 'W') + cvrt(pin_m2, 'dBm', 'W') + cvrt(pin_m3, 'dBm', 'W')).*100;
	
	% Save data
	CE_data(idx) = {pchip_mCE};
end


%% Plot Chip-Plane data

lw = 1.5;
mks = 10;

LEs = {};

figure(1);
subplot(1, 1, 1);
hold off;
idx = 0;
cm_offset = 1;
CM = resamplecmap('parula', numel(power_levels)+cm_offset);
for pwr = power_levels
	idx = idx + 1;
	
	data = CE_data{idx};
	I_outlier = (data > 20);
	
	plot(Vdcs(~I_outlier).*iv_conv.*1e3, data(~I_outlier), 'Marker', '.', 'MarkerSize', 12, 'LineStyle', ':', 'LineWidth', 1.5, 'Color', CM(idx, :));
	hold on;
	LEs{idx} = "P_{RF} = "+num2str(pwr) + " dBm";
end
grid on;
legend(LEs{:});
xlabel("DC Bias Current (mA)");
ylabel("Conversion Efficiency (%)");
title("Conversion Efficiency Power Dependence");
