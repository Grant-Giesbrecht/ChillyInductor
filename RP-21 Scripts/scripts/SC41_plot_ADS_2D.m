ads_path = '/Volumes/M4 PHD/ARC0 PhD Data/RP-21 Kinetic Inductance 2023/Data/group4_extflash/ADS Sim Data/';

% [harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D("norm_100K_2Dsweep.csv");
% [harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D("fine_100K_2Dsweep.csv");
% [harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D("5_15GHz_ultra_sim.csv");
% [harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D("1_5GHz_ultra_sim.csv");

% [harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D(fullfile(ads_path, "ADS_10GHz_Ultrafine.csv"));

% NOTE: freq_SG and harm_idx are flipped for this format! - nope this
% doesn't have enough sweeps - this file can't work in this script.
[freq_SG, harm_idx, Vbias, Pload_mag, Pload_deg] = load_ADS_2D(fullfile(ads_path, "Wide_FreqSweep_100K_ForPub.csv"));

max_allowed_bias = 1.5;

% Get unique values
freqs = (unique(freq_SG));
biases = unique(Vbias);
harms = unique(harm_idx);

% Create grid
[B, F] = meshgrid(biases, freqs);

% Create harmonic masks
Ih1 = (harm_idx == 1);
Ih2 = (harm_idx == 2);
Ih3 = (harm_idx == 3);

% Create power variable
V_h1 = zeros(size(F));
V_h2 = zeros(size(F));
V_h3 = zeros(size(F));

idx_f = 0;
for f = freqs
	idx_f = idx_f + 1;
	
	% Create frequency mask
	Iw = (freq_SG == f);
	
	idx_b = 0;
	for b = biases
		idx_b = idx_b + 1;
		
		% Create bias and sub masks
		Ib = (Vbias == b);
		submask = Iw & Ib;
		
		% Get fundamental
		p1 = Pload_mag(submask & Ih1);
		if numel(p1) ~= 1
			warning("Wrong number of points");
		end
		V_h1(idx_f, idx_b) = p1;
		
		p2 = Pload_mag(submask & Ih2);
		V_h2(idx_f, idx_b) = p2;
		
		p3 = Pload_mag(submask & Ih3);
		V_h3(idx_f, idx_b) = p3;
		
	end
end

Z0 = 50;
P_h1 = V_h1.^2./Z0;
P_h2 = V_h2.^2./Z0;
P_h3 = V_h3.^2./Z0;

CE2 = 100.*P_h2./(P_h1 + P_h2 + P_h3);
CE2sys = 100.*P_h2./cvrt(4, 'dBm', 'W');

figure(1);
surf(B, F, P_h1);
ylabel("Frequency (GHz)");
xlabel("Bias Voltage (V)");
zlabel("Power (W)");
title("ADS Simulation: Fundamental Power");
grid on;

figure(2);
surf(B, F, P_h2);
ylabel("Frequency (GHz)");
xlabel("Bias Voltage (V)");
zlabel("Power (W)");
title("ADS Simulation: 2nd Harmonic Power");
grid on;

figure(3);
surf(B, F, P_h3);
ylabel("Frequency (GHz)");
xlabel("Bias Voltage (V)");
zlabel("Power (W)");
title("ADS Simulation: 3rd Harmonic Power");
grid on;

figure(4);
hold off
surf(B, F, CE2);
hold on
ylabel("Frequency (GHz)");
xlabel("Bias Voltage (V)");
zlabel("Chip Conversion Efficiency (%)");
title("ADS Simulation: 2nd Harm. Conversion Effic.");
grid on;

figure(6);
hold off
surf(B, F, CE2sys);
hold on
ylabel("Frequency (GHz)");
xlabel("Bias Voltage (V)");
zlabel("System Conversion Efficiency (%)");
title("ADS Simulation: 2nd Harm. Conversion Effic.");
grid on;

%% Generate max CE over Freq plot

Ibm = (Vbias <= max_allowed_bias);

CE2_max = zeros(1, numel(freqs));
CE2_max_Vb = zeros(1, numel(freqs));
idx_f = 0;
for f = freqs
	idx_f = idx_f + 1;
	
	% Create frequency mask
	Iw = (freq_SG == f);
	
	% Calculate power at each bias point
	p1 = Pload_mag(Iw & Ibm & Ih1).^2./Z0;
	p2 = Pload_mag(Iw & Ibm  & Ih2).^2./Z0;
	p3 = Pload_mag(Iw & Ibm  & Ih3).^2./Z0;
	biases = Vbias(Iw & Ibm  & Ih1);
	
	% Calculate CE
	CE2_allbias = p2./(p1+p2+p3).*100;
	
	% Save maximum
	[CE2_max(idx_f), idx_max] = max(CE2_allbias);
	
	% Save corresponding bias
	CE2_max_Vb(idx_f) = biases(idx_max);
end

figure(3);
plot3(CE2_max_Vb, freqs, CE2_max, 'LineStyle', '--', 'Marker', '.', 'Color', [0, 0, 0.65], 'MarkerSize', 13);

figure(5);
hold off;
plot(freqs, CE2_max, 'LineStyle', ':', 'Marker', '.', 'Color', [0, 0, 0.65], 'MarkerSize', 13);
grid on;
xlabel("Frequency (GHz)");
ylabel("Maximum Conversion Efficiency (%)");
title("2nd Harmonic Conversion Efficiency");

return;
%% Zoom in on specific region

match_axes_of_fig = 1; % Figure whose axes to match

figure(match_axes_of_fig);
xl_ = xlim();
yl_ = ylim();
zl_ = zlim();

figure(3);
xlim(xl_);
ylim(yl_);
zlim(zl_);




