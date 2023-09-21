%% SC29
%
% Previously named DS52
%
% Accepts an alpha sweep and shows frequency fundamental sweep and compares
% against ADS simulation results.
%

DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');

load(fullfile(DATA_PATH, 'alpha_12Sept2023.mat'));

% Options, -6, 0, 4, 6 dBm
power = 4;

% Options = 0, 2
bias = 0;

% Get all frequencies
freqs = unique([ds.dataset.SG_freq_Hz]);
powers = [ds.dataset.SG_power_dBm];
biases = [ds.dataset.offset_V];
freq_points = [ds.dataset.SG_freq_Hz];

I_pwr = (powers == power);
I_bias = (biases == bias);

power_dBm = zeros(1, numel(freqs));

idx_ = 1;
for f = freqs
	
	% Generate mask
	I_freq = (freq_points == f);
	mask = I_pwr & I_bias & I_freq;
	
	% Scan over all matching points
	pts = [ds.dataset(mask)];
	pwr_dBm = zeros(1, numel(pts));
	idx = 0;
	for pt = pts
		idx = idx + 1;
		pwr_dBm(idx) = pt.SG_power_dBm - abs(VNA2dBm(mean(pt.VNA_data.data(1,:))));
	end
	
	power_dBm(idx_) = mean(pwr_dBm);
	
	idx_ = idx_ + 1;
end

figure(1);
hold off;
plot(freqs./1e9, power_dBm, 'LineStyle', '--', 'Color', [0, 0, 0.7], 'Marker', 'o');
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dB)");
title("Fundamental Transmission Coefficient");

