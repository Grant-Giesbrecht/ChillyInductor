%% Repeat of SC17 to analyze FF1 dataset and produce graph from paper draft

split_freqs = [15e9, 25e9];

% Plot Measured Data

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

%% Repeat of SC29 to analyze V2+ datasets of 'alpha' configuration

DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
DATA_PATH_SUPP = fullfile('/','Volumes', 'NO NAME','NIST Datasets Supplementary','group3_supplementary');
DATA_PATH_EXT = fullfile('/','Volumes', 'NO NAME','NIST September data');

warning off;
% load(fullfile(DATA_PATH, 'alpha_14Sept2023_1600.mat'));
% load(fullfile(DATA_PATH, 'alpha2_20Sept2023_3K.mat'));
% load(fullfile(DATA_PATH_EXT, 'alpha3_17Oct2023.mat'));
% load(fullfile(DATA_PATH_EXT, 'alpha3_18Oct2023_CalCALREG.mat'));
load(fullfile(DATA_PATH_EXT, 'alpha0BW_18Oct2023_CalCryoAug23.mat'));

warning on;

% Options, -6, 0, 4, 6 dBm
power = 0;

% Options = 0, 2
bias = 0;

% Get all frequencies
freqs_ = unique([ds.dataset.SG_freq_Hz]);
powers = [ds.dataset.SG_power_dBm];
biases = [ds.dataset.offset_V];
freq_points = [ds.dataset.SG_freq_Hz];

I_pwr = (powers == power);
I_bias = (biases == bias);

power_dBm = zeros(1, numel(freqs_));

idx_ = 1;
for f = freqs_
	
	% Generate mask
	I_freq = (freq_points == f);
	mask = I_pwr & I_bias & I_freq;
	
	% Scan over all matching points
	pts = [ds.dataset(mask)];
	pwr_dBm = zeros(1, numel(pts));
	idx = 0;
	for pt = pts
		idx = idx + 1;
		pwr_dBm(idx) = pt.SG_power_dBm - abs(VNA2dBm(mean(abs(pt.VNA_data.data(1,:)))));
	end
	
	power_dBm(idx_) = mean(pwr_dBm);
	
	idx_ = idx_ + 1;
end

%% Plot results

mkz1 = 15;
blue1 = [0, 0, 0.65];
green1 = [0, 0.65, 0];
% green1 = [0.7, 0.4, 0];
% green1 = [0.7, 0, 0];
% green1 = [0, 0.4, 0.7];

figure(1);
% hold off;
% plot(freqs(I1)./1e9, lin2dB(avg_S21(I1)), 'Marker', '.', 'MarkerSize', mkz1, 'Color', blue1, 'LineStyle', ':');
hold on;
plot(freqs_./1e9, power_dBm, 'Marker', '.', 'MarkerSize', mkz1, 'Color', green1, 'LineStyle', ':');
xlim([9.8, 10.2]);
grid on;
xlabel("Frequecy (GHz)");
ylabel("S_{21}");
title("Comparison of Resonance Behavior");
legend("Basic 'Warm' Cal", "4K Cal");































