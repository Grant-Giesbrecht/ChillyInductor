
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

c_loss = [0.5, 0, 0.6];
c_sim = [0.7, 0, 0];
c_scaled = [0, .6, 0];
c_meas = [0, 0, .6];

figure(3);
hold off;
% plot(freqs(I1)./1e9, avg_S21(I1).*a_SG, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
plot(freqs(I1)./1e9, cvrt(b1(I1).^2, 'W', 'dBm'), 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
% plot(freqs(I1)./1e9, avg_S21(I1), 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2);
hold on;
plot(freq_GHz, Pload_scaled+6, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 10);
plot(freq_GHz, Pload_dBm, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 10);
plot(freq_GHz, S21_applied, 'LineStyle', '--', 'Marker', '+', 'LineWidth', 1.2, 'MarkerSize', 10);
grid on;
xlabel("Frequency (GHz)");
ylabel("Mixed")
title("Invalid comparison - I vs P(dBm)")
force0y;
legend("Measurement", "ADS Simulation, Loss Applied, 6 dBm Shift", "ADS Simulation, No Loss", "Measured Loss");

figure(1);
subplot(3, 1, 1);
hold off;
plot(freq_GHz, Pload_dBm, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 10, 'Color', c_sim);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dBm)")
title("Simulation Output")


subplot(3, 1, 2);
hold off;
plot(freq_GHz, S21_applied, 'LineStyle', '--', 'Marker', '+', 'LineWidth', 1.2, 'MarkerSize', 10, 'Color', c_loss);
hold on;
scatter(loss_freqs, loss_vals, loss_vals.*0+30, 'MarkerFaceColor', [0    0.4470    0.7410]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dBm)")
title("Measured System Loss")
legend("Measurement", "Interpolation Result");

subplot(3, 1, 3);
hold off;
plot(freqs(I1)./1e9, cvrt(b1(I1).^2, 'W', 'dBm')-6, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2, 'Color', c_meas);
hold on;
plot(freq_GHz, Pload_scaled, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 10, 'Color', c_scaled);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dBm)")
title("Simulation versus Measurement, Inlcuding Measured Loss")
legend("Measurement", "ADS Simulation");
xlim([9.8, 10.2]);

return;


plot(freq_GHz, Pload_dBm, 'LineStyle', '-.', 'Marker', 'o', 'LineWidth', 1.2, 'MarkerSize', 10);

grid on;
xlabel("Frequency (GHz)");
ylabel("Mixed")
title("Invalid comparison - I vs P(dBm)")
force0y;
legend("Measurement", "ADS Simulation, Loss Applied, 6 dBm Shift", "ADS Simulation, No Loss", "Measured Loss");

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
subplot(1, 3, 3);
xlim(band);


return









