%% SC27
%
% Previously named: DS50
%
% Accepts a set of S2P files and calculates the distributed reactance of
% the line, including loss (if S21 is given). Written in NC <3

%% Configure data input

%----------------------------- FILE DATA -------------------------------

% % Chip1 at 4,0K 4 dBm
% DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '4K', 'LC 4K 4dBm');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
% EXTENSION = "_trimmed.s2p";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

% % Chip1 at 4,0K -10 dBm
% DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '4K', 'LC 4K -10dBm');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2", "1V4"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4];
% EXTENSION = "_trimmed.s2p";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

% Chip1 at 3,0K, 4 dBm
DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '3K', 'LC 3K0 4dBm');
FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2"];
bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
EXTENSION = "_trimmed.s2p";

S11_PREFIX = "LC_S11_";
S21_PREFIX = "LC_S21_";

% % Chip1 at 3,0K, -10 dBm
% DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep', '3K', 'LC 3K0 -10dBm');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V2", "1V4", "1V6"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6];
% EXTENSION = "_trimmed.s2p";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

% % Chip1 at 2,88K
% DATA_PATH = fullfile('/','Volumes','NO NAME','October Temperature Sweep','LC 2,88K -10dBm');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V1", "1V2", "1V4"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4];
% EXTENSION = ".prn";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

% % Chip2 at 4K
% DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Chip_r3c2_LC');
% FILE_POSTFIXES = ["0V0", "0V2", "0V4", "0V6", "0V8", "1V0", "1V1", "1V2"];
% bias_voltage = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2];
% EXTENSION = ".s2p";
% 
% S11_PREFIX = "LC_S11_";
% S21_PREFIX = "LC_S21_";

%----------------------------- NULL DATA ------------------------------

NULL_NUM = 4; % Which null is being targeted
NULL_FREQ_LOW_HZ = 300e6; % Frequency bounds to target
NULL_FREQ_HIGH_HZ = 400e6;

% Lower and upper limits of bands in which to find detuned gamma magnitude
DETUNE_FREQ_MIN_HZ = [100e6, 187e6, 276e6];
DETUNE_FREQ_MAX_HZ = [156e6, 237e6, 328e6];

%--------------------------- SYSTEM DATA --------------------------------

Z0 = 50;
l_phys = 0.5;

%---------------------------- MODEL OPTIONS ----------------------------

USE_LOSS_MODEL = false;
USE_ALT_PLOTTING = false;

clr1 = [0, 0, 0.6];
clr2 = [0, 0.6, 0];
clr3 = [0.7, 0, 0];

mk_s = 'o';
mk_alt = '+';
standard_label = "No Loss Model";
alt_label = "With S_{21} Loss Model";

%% Process files

figure(1);
subplot(1, 2, 1);
hold off;
subplot(1, 2, 2);
hold off;

if ~USE_ALT_PLOTTING
	figure(2);
	hold off;
	figure(3);
	hold off;
	figure(4);
	hold off;
end

% Iterate over files
nf = numel(FILE_POSTFIXES);
RCM = resamplecmap('winter', nf);
Ls = zeros(1, nf);
Cs = zeros(1, nf);
Zcs = zeros(1, nf);
for idx = 1:nf
	
	% Read files
	S11_raw = sparameters(fullfile(DATA_PATH, S11_PREFIX+FILE_POSTFIXES(idx)+EXTENSION));
	S21_raw = sparameters(fullfile(DATA_PATH, S21_PREFIX+FILE_POSTFIXES(idx)+EXTENSION));
	S11_dB = lin2dB(abs(flatten(S11_raw.Parameters(1, 1, :))));
	S21_dB = lin2dB(abs(flatten(S11_raw.Parameters(2, 1, :))));
	
	% Add to figure 1
	figure(1);
	subplot(1, 2, 1);
	plot(S11_raw.Frequencies./1e6, S11_dB, 'Color', RCM(idx, :));
	hold on;
	subplot(1, 2, 2);
	plot(S21_raw.Frequencies./1e6, S21_dB, 'Color', RCM(idx, :));
	hold on;
	
	% Get frequency of selected null
	mask = (S11_raw.Frequencies >= NULL_FREQ_LOW_HZ) & (S11_raw.Frequencies <= NULL_FREQ_HIGH_HZ);
	[val, fm_idx] = min(S11_dB(mask));
	masked_freqs = S11_raw.Frequencies(mask);
	null_freq = masked_freqs(fm_idx);
	
	% Get reflection magnitude in each bin
	gammas = zeros(1, numel(DETUNE_FREQ_MAX_HZ));
	for midx = 1:numel(DETUNE_FREQ_MAX_HZ)
		
		% Apply mask for range
		mask = (S11_raw.Frequencies >= DETUNE_FREQ_MIN_HZ(midx)) & (S11_raw.Frequencies <= DETUNE_FREQ_MAX_HZ(midx));
		
		% Find detune/max
		[detune_S11, fm_idx] = max(S11_dB(mask));
		
		% Calc. masked parameters
		masked_freqs = S11_raw.Frequencies(mask);
		masked_S21 = S21_dB(mask);
		
		% Calculate S11 and S21 at max/detuned value
		detune_freq = masked_freqs(fm_idx);
		S11m = dB2lin(detune_S11);
		S21m = dB2lin(masked_S21(fm_idx));
		
		% Estimate reflection coefficient
		if USE_LOSS_MODEL
			
			delta = S11m^2 + S21m^2;
			L = delta.^0.25;
			R = S11m./L^2;
			
			displ("R: ", lin2dB(R), " detune_S11: ", detune_S11)
			gammas(midx) = lin2dB(R);
			
		else
			gammas(midx) = detune_S11;
		end
		
		
	end
	G = dB2lin(mean(gammas));
	
	% Calculate L and C
	L = NULL_NUM.*Z0./(2 .*null_freq.*l_phys).*sqrt((1 + G)./(1 - G));
	C = (NULL_NUM./(2.*l_phys.*null_freq.*sqrt(L))).^2;
	Zchip = Z0.*sqrt((1+G)./(1-G));
	
	% Save data
	Ls(idx) = L;
	Cs(idx) = C;
	Zcs(idx) = Zchip;
	
end

% Format figure 1
subplot(1, 2, 1);
grid on;
xlabel("Frequnecy (MHz)");
ylabel("S_{11} (dB)");
title("S_{11} Bias Sweep");
ylim([-55, 0]);
subplot(1, 2, 2);
grid on;
xlabel("Frequnecy (MHz)");
ylabel("S_{21} (dB)");
title("S_{21} Bias Sweep");
ylim([-2, 0]);

mkz = 8;
lw = 1.5;

figure(2);
if USE_ALT_PLOTTING
	hold on;
	plot(bias_voltage, Cs.*1e12, 'LineStyle', ':', 'Marker', mk_alt, 'Color', clr1, 'LineWidth', lw, 'MarkerSize', mkz);
else
	plot(bias_voltage, Cs.*1e12, 'LineStyle', ':', 'Marker', mk_s, 'Color', clr1, 'LineWidth', lw, 'MarkerSize', mkz);
end
grid on;
xlabel("Bias Voltage (V)");
ylabel("Distributed Capacitance (pF/m)");
title("Distributed Capacitance over Bias");
% ylim([148, 153]);
% yticks(146:1:155);

% fix limits
C_avg = round(mean(Cs.*1e12));
ylim([C_avg-1, C_avg+1]);
yticks((C_avg-1):0.25:(C_avg+1));

figure(3);
if USE_ALT_PLOTTING
	hold on;
	plot(bias_voltage, Ls.*1e9, 'LineStyle', ':', 'Marker', mk_alt, 'Color', clr2, 'LineWidth', lw, 'MarkerSize', mkz);
else
	plot(bias_voltage, Ls.*1e9, 'LineStyle', ':', 'Marker', mk_s, 'Color', clr2, 'LineWidth', lw, 'MarkerSize', mkz);
end
grid on;
xlabel("Bias Voltage (V)");
ylabel("Distributed Inductance (nH/m)");
title("Distributed Inductance over Bias");

figure(4);
if USE_ALT_PLOTTING
	hold on;
	plot(bias_voltage, Zcs, 'LineStyle', ':', 'Marker', mk_alt, 'Color', clr3, 'LineWidth', lw, 'MarkerSize', mkz);
else
	plot(bias_voltage, Zcs, 'LineStyle', ':', 'Marker', mk_s, 'Color', clr3, 'LineWidth', lw, 'MarkerSize', mkz);
end
grid on;
xlabel("Bias Voltage (V)");
ylabel("Characteristic Impedance (Ohms)");
title("Characteristic Impedance vs Bias");

if USE_ALT_PLOTTING
	figure(2);
	legend(standard_label, alt_label);
	figure(3);
	legend(standard_label, alt_label);
	figure(4);
	legend(standard_label, alt_label);
end

