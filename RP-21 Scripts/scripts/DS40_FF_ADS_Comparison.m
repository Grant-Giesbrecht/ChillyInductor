
USE_LOSS = false;

%% CONTROL
% Discrete sections at bottom allow alteration of data zoom.

%% Set Conditions

split_freqs = [15e9, 25e9];

%% Plot Measured Data

load(dataset_path("ads_sweep5.mat"));

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
std_S21 = zeros(1, numel(freqs));

% Get mean of VNA data for b1a2
for idx = 1:numel(S21)
	
	% Get mean of data (each point in ld array contains a whole sweep)
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

I1 = (freqs < split_freqs(1));
I2 = (freqs < split_freqs(2)) & (freqs > split_freqs(1));
I3 = (freqs > split_freqs(2));

%% Apply loss data to ADS results

% Duplicate power data
Pload_scaled = Pload_dBm;

if USE_LOSS
	% Look up closest S21 measurement for each point, and scale appropriately
	idx = 0;
	for f = freq_GHz
		idx = idx+1;
		find_idx = findClosest(freq_Hz, f*1e9);

		Pload_scaled(idx) = Pload_scaled(idx)+S21_dB(idx);	
	end
end

S21_ads = Pload_scaled - 6; % Dataset used 6 dBm of input power

%% Plot results

figure(1);
hold off;
% plot(freqs(I1)./1e9, Iac(I1).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
plot(freqs(I1)./1e9, avg_S21(I1), 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
hold on;
plot(freq_GHz, S21_ads, 'LineStyle', '-.', 'Marker', '+', 'LineWidth', 1.2);
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Current Measurement over Frequency - Band 1")
force0y;

% figure(2);
% hold off;
% plot(freqs(I2)./1e9, Iac(I2).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
% hold on;
% grid on;
% xlabel("Frequency (GHz)");
% ylabel("VNA Current (mA)")
% title("Current Measurement over Frequency - Band 2")
% force0y;
% 
% figure(3);
% hold off;
% plot(freqs(I3)./1e9, Iac(I3).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
% hold on;
% grid on;
% xlabel("Frequency (GHz)");
% ylabel("VNA Current (mA)")
% title("Current Measurement over Frequency - Band 3")
% force0y;

return

%% Zoom auto

figure(1);
xlim('auto');
figure(2);
xlim('auto');
figure(3);
xlim('auto');

%% Zoom center - const BW

band = [9.8, 10.2]; 
figure(1);
xlim(band);

figure(2);
xlim(band+10);

figure(3);
xlim(band+20);

return









