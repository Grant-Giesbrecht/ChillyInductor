%% Merger of DS42 and DS43

%% DS42 Copy

USE_LOSS = true;
INTERPOLATE_LOSS = true;

%% CONTROL
% Discrete sections at bottom allow alteration of data zoom.

%% Set Conditions

split_freqs = [15e9, 25e9];

%% Plot Measured Data

load(dataset_path("ads_sweep6.mat"));
RFpwr_simulation_dBm = 0;

load(dataset_path("cryostat_sparams.mat"));

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
b1 = zeros(1, numel(freqs));
std_S21 = zeros(1, numel(freqs));

% Get mean of VNA data for b1a2
for idx = 1:numel(S21)
	
	% Get mean of data (each point in ld array contains a whole sweep)
	b1a2 = mean(ld(idx).VNA_data.data(1,:));

	% Convert a1b2 to S21
	b1(idx) = abs(b1a2).*a2;
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

I1 = (freqs < split_freqs(1));
I2 = (freqs < split_freqs(2)) & (freqs > split_freqs(1));
I3 = (freqs > split_freqs(2));

%% Apply loss data to ADS results

% Duplicate power data
Pload_scaled = Pload_dBm;

S21_applied = Pload_dBm;

if USE_LOSS
	
	displ("Use loss!");
	
	% Look up closest S21 measurement for each point, and scale appropriately
	idx = 0;
	for f = freq_GHz
		idx = idx+1;
		find_idx = findClosest(freq_Hz, f*1e9);
		
		if INTERPOLATE_LOSS
			S21_applied(idx) = interp1(freq_Hz, S21_dB, f*1e9);
			Pload_scaled(idx) = Pload_scaled(idx)+S21_applied(idx);
		else
			S21_applied(idx) = S21_dB(find_idx);
			Pload_scaled(idx) = Pload_scaled(idx)+S21_applied(idx);
		end
		
		
	end
end

S21_ads = Pload_scaled - RFpwr_simulation_dBm; % Dataset used 6 dBm of input power

loss_idxs = (freq_Hz >= 9.8e9) & (freq_Hz <= 10.2e9);
loss_freqs = freq_Hz(loss_idxs)./1e9;
loss_vals = S21_dB(loss_idxs);


%% Plot results

c_ads = [0.7, 0, 0];
c_meas = [0, 0, .6];

% lw_meas = 1;
% lw_ads = 2;
% mkz_meas = 25;
% mkz_ads = 10;

lw_meas = 0.5;
lw_ads = 1;
mkz_meas = 12;
mkz_ads = 5;


figure(1);
subplot(1, 2, 1);
hold off;
plot(freqs(I1)./1e9, cvrt(b1(I1).^2, 'W', 'dBm')-6, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw_meas, 'Color', c_meas, 'MarkerSize', mkz_meas);
hold on;
plot(freq_GHz, Pload_scaled, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', lw_ads, 'MarkerSize', mkz_ads, 'Color', c_ads);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dBm)")
title("Simulation versus Measurement, Inlcuding Measured Loss")
legend("Measurement", "ADS Simulation");
xlim([9.8, 10.2]);

%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

load(dataset_path("ads_sweep4_phase.mat"));

load(dataset_path("cryostat_sparams.mat"));

%% DS43 Copy

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
subplot(1, 2, 2);
hold off;
% plot(Vdcs, theta-theta(floor(numel(theta)/2)+1), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
% hold on;
% plot(Vdc, phase_deg-phase_deg(1), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
plot(Vdcs, theta-theta(floor(numel(theta)/2)+1), 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw_meas, 'Color', c_meas, 'MarkerSize', mkz_meas);
hold on;
plot(Vdc, phase_deg-phase_deg(1), 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', lw_ads, 'MarkerSize', mkz_ads, 'Color', c_ads);
grid on;
legend("Measured", "ADS Simulation");
xlabel("DC Bias Voltage (V)");
ylabel("Phase (^\circ)");
title("ADS vs Measurement Phase Comparison, P_{RF} = 4 dBm");
xlim([0, 3]);








