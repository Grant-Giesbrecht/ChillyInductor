%% Dateinamen Angeben

dividers = [.05, 2:2:10, 15:5:40, 50].*1e9;
div_str = "50MHz";

for d_idx = 2:numel(dividers)
	div_str(d_idx) = num2str(dividers(d_idx)./1e9)+"GHz";
end

S11_prefix = "S11_";
S21_prefix = "S21_";
postfix = "_trimmed.s2p";

if ispc
	datapath = fullfile("E:\ARC0 PhD Data\RP-21 Kinetic Inductance 2023\Data\group4_extflash\S21 Cal Verification", "8_Nov_Compiled_S21");
else
	datapath = fullfile("/", "Volumes", "M4 PHD", "ARC0 PhD Data", "RP-21 Kinetic Inductance 2023", "Data", "group4_extflash", "S21 Cal Verification", "8_Nov_Compiled_S21");
end
%% Alle Dateien lesen

filename_11 = fullfile(datapath, S11_prefix + div_str(1) + "_" + div_str(2) + postfix);
filename_21 = fullfile(datapath, S21_prefix + div_str(1) + "_" + div_str(2) + postfix);

S11_temp = sparameters(filename_11);
S21_temp = sparameters(filename_21);

S11_meister_freq_Hz = S11_temp.Frequencies;
S21_meister_freq_Hz = S21_temp.Frequencies;
S11_meister_sparam = S11_temp.Parameters;
S21_meister_sparam = S21_temp.Parameters;

% Schleife über alle Dateien
for idx = 2:numel(dividers)-1
	
	% Generate file names
	filename_11 = fullfile(datapath, S11_prefix + div_str(idx) + "_" + div_str(idx+1) + postfix);
	filename_21 = fullfile(datapath, S21_prefix + div_str(idx) + "_" + div_str(idx+1) + postfix);
	
	% Read data files
	S11_temp = sparameters(filename_11);
	S21_temp = sparameters(filename_21);
	
	% Merge data into master - SParameters
	S11_meister_sparam = cat(3, S11_meister_sparam, S11_temp.Parameters(:, :, 2:end));
	S21_meister_sparam = cat(3, S21_meister_sparam, S21_temp.Parameters(:, :, 2:end));
	
	% Merge data into master - Frequencies
	S11_meister_freq_Hz = [S11_meister_freq_Hz; S11_temp.Frequencies(2:end)];
	S21_meister_freq_Hz = [S21_meister_freq_Hz; S21_temp.Frequencies(2:end)];
	
end


%% Plot results

S21_dB = flatten(lin2dB(abs(S21_meister_sparam(2, 1, :))));
S11_dB = flatten(lin2dB(abs(S11_meister_sparam(1, 1, :))));

figure(1);
plot(S21_meister_freq_Hz'./1e9, S21_dB, 'LineStyle', ':', 'Marker', '.', 'Color', [68, 217, 240]./255);
hold on;
plot(S11_meister_freq_Hz'./1e9, S11_dB, 'LineStyle', ':', 'Marker', '.', 'Color', [240, 220, 68]./255);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} and S_{21} (dB)");
title("S-Parameters of Chip");
legend("S_{21}", "S_{11}", 'Location', 'Best');
ylim([-40, 0]);

figure(2);
subplot(2, 1, 1);
plot(S21_meister_freq_Hz'./1e9, S21_dB, 'LineStyle', ':', 'Marker', '.', 'Color', [68, 217, 240]./255);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dB)");
title("S_{21} of Chip with Calibration");
ylim([-40, 0]);

subplot(2, 1, 2);
plot(S11_meister_freq_Hz'./1e9, S11_dB, 'LineStyle', ':', 'Marker', '.', 'Color', [240, 220, 68]./255);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} (dB)");
title("S_{11} of Chip with Calibration");
ylim([-40, 0]);









