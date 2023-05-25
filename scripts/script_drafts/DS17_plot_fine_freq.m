


% Import data
load(dataset_path("DS6_10GHzFreqSweep_FF1.mat"));
% load("C:\Users\Grant Giesbrecht\Downloads\FF1.mat");

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
V = S21.*a_SG.*sqrt(50);
Iac = V./50;

figure(3);
hold off;
scatter(all_freq./1e9, Iac.*1e3, 'Marker', '+');
grid on;
xlabel("Frequency (Hz)");
ylabel("Expected VNA Current (mA)")
title("Current at Fundamental")
force0y;

figure(4);
hold off;
plot(freqs./1e9, avg_S21.*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
grid on;
xlabel("Frequency (Hz)");
ylabel("Expected VNA Current (mA)")
title("Averaged Current at Fundamental")
force0y;