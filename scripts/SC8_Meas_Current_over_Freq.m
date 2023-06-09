% Previously DS13 fund current over freq
%
% Looks at how the current at the fundamental evolves over frequency from
% the FOS-1 dataset.

% Load data
load(dataset_path("DS1o1_FOS-1A.mat"));

% Filter all points with zero bias (this is half of all points!)
I1 = (Vbias == 0);
freqs = unique(freq);

% For each frequency, find average and stdev
stdevs = zeros(1, numel(freqs));
averages = zeros(1, numel(freqs));
idx = 0;
for f = freqs
	idx = idx + 1;
	
	% Find all points with frequency 'f'
	I2 = (freq(I1) == f);
	
	% Calculate stats
	Vs = dB2lin(Mb1a2(I2));
	averages(idx) = mean(Vs);
	stdevs(idx) = std(Vs);
	
end


% Approximate Ipp from fundamental
a_SG = sqrt(cvrt(SG_power_dBm, 'dBm', 'W')); % There is only one power used
a2 = sqrt(cvrt(-10, 'dBm', 'W')); % VNA port 1 power
S21_fund = abs(averages).*a2./a_SG;
V_port_fund = S21_fund.*a_SG.*sqrt(50);
Ipp_fund = V_port_fund./50;

%% Generate Plots

figure(1);
plot(freqs./1e9, averages, 'Marker', '+', 'LineStyle', ':', 'Color', [0, 0, .7]);
force0y;
grid on;
xlabel("Frequency (GHz)");
ylabel("Current (A)");
title("Fundamental Current over Frequency");