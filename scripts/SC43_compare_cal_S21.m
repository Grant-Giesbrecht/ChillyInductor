s2p_dpath = '/Volumes/M6 T7S/ARC0 PhD Data/RP-21 Kinetic Inductance 2023/Data/group4_extflash/S21 Cal Verification/Warm_Cal_S21';

file_nocal = fullfile(s2p_dpath, "SP_50MHz_50GHz_NoCal_trim.s2p");
file_cryocal = fullfile(s2p_dpath, "SP_50MHz_50GHz_cryo2_trim.s2p");

FIG1 = 10;
FIG2 = 20;
FIG3 = 11;

sc = sparameters(file_cryocal);
sn = sparameters(file_nocal);

%% Generate Plots

figure(FIG1);
subplot(2, 2, 1);
hold off;
plot(sc.Frequencies./1e9, flatten(lin2dB(abs(sc.Parameters(2, 1, :)))), 'LineStyle', ':', 'Marker', '.');
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dB)");
title("S_{21}, Cryostat Calibration Applied");
ylim([-20, 0]);
xlim([0, 50]);

subplot(2, 2, 3);
hold off;
plot(sc.Frequencies./1e9, flatten(lin2dB(abs(sc.Parameters(1, 1, :)))), 'LineStyle', ':', 'Marker', '.', 'Color', [0.7, 0.2, 0]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} (dB)");
title("S_{11}, Cryostat Calibration Applied");
ylim([-20, 10]);
xlim([0, 50]);

figure(FIG2);
subplot(2, 2, 1);
hold off;
plot(sn.Frequencies./1e9, flatten(lin2dB(abs(sn.Parameters(2, 1, :)))), 'LineStyle', ':', 'Marker', '.');
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{21} (dB)");
title("S_{21}, No Calibration Applied");
ylim([-20, 0]);
xlim([0, 50]);

subplot(2, 2, 3);
hold off;
plot(sn.Frequencies./1e9, flatten(lin2dB(abs(sn.Parameters(1, 1, :)))), 'LineStyle', ':', 'Marker', '.', 'Color', [0.7, 0.2, 0]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} (dB)");
title("S_{11}, No Calibration Applied");
ylim([-20, 10]);
xlim([0, 50]);

total_cryo = dBsum(flatten(lin2dB(abs(sc.Parameters(1, 1, :)))), flatten(lin2dB(abs(sc.Parameters(2, 1, :)))));
total_nocal = dBsum(flatten(lin2dB(abs(sn.Parameters(1, 1, :)))), flatten(lin2dB(abs(sn.Parameters(2, 1, :)))));

figure(FIG1);
subplot(2, 2, [2,4]);
hold off;
plot(sc.Frequencies./1e9, total_cryo, 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} + S_{21} (dB)");
title("S_{11} + S_{21}, Cryostat Calibration Applied");
ylim([-20, 10]);
xlim([0, 50]);

figure(FIG2);
subplot(2, 2, [2,4]);
hold off;
plot(sc.Frequencies./1e9, total_nocal, 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} + S_{21} (dB)");
title("S_{11} + S_{21}, No Calibration Applied");
ylim([-20, 10]);
xlim([0, 50]);

%% Mean Lines

mlw = 1;

cm_lin = mean(dB2lin(total_cryo));
cs_lin = std(dB2lin(total_cryo));

cm = cm_lin;
cs_l = cm_lin-cs_lin;
cs_h = cm_lin+cs_lin;

% cm_dB = lin2dB(cm_lin);
% cs_l = lin2dB(cm_lin - cs_lin);
% cs_h = lin2dB(cm_lin + cs_lin);

figure(FIG3);
subplot(1, 1, 1);
hold off;
plot(sc.Frequencies./1e9, dB2lin(total_cryo), 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
hold on;
Lm1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cm, cm], 'LineStyle', '--', 'Color', [0, 0, 0], 'LineWidth', mlw);
Ls1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_l, cs_l], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
Ls2 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_h, cs_h], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
grid on;
xlabel("Frequency (GHz)");
ylabel("S_{11} + S_{21}, Linear");
title("Statistical Significance, Linear Scale");
legend("S_{11} + S_{21}", "Mean", "Standard Deviation");

ylim([-10, 10]);

%% Universal X limit

XL = [0, 50];
YL = [-6, 6];

for fn = [FIG1, FIG2]
	
	figure(fn);
	
	subplot(2, 2, 1);
	xlim(XL);
	subplot(2, 2, 3);
	xlim(XL);
	subplot(2, 2, [2,4]);
	xlim(XL);
	ylim(YL);
	
end

% Recalcualte means

delete(Lm1);
delete(Ls1);
delete(Ls2);

I = (sc.Frequencies./1e9 >= XL(1)) & (sc.Frequencies./1e9 <= XL(2));

cm_lin = mean(dB2lin(total_cryo(I)));
cs_lin = std(dB2lin(total_cryo(I)));

cm = cm_lin;
cs_l = cm_lin-cs_lin;
cs_h = cm_lin+cs_lin;

if cs_l > 0
	figure(FIG3);
	subplot(1, 2, 1);
	hold off;
	plot(sc.Frequencies./1e9, dB2lin(total_cryo), 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
	hold on;	
	Lm1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cm, cm], 'LineStyle', '--', 'Color', [0, 0, 0], 'LineWidth', mlw);
	Ls1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_l, cs_l], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	Ls2 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_h, cs_h], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	grid on;
	xlabel("Frequency (GHz)");
	ylabel("S_{11} + S_{21}, Linear");
	title("Statistical Significance, Linear Scale");
	legend("S_{11} + S_{21}", "Mean", "Standard Deviation");
	xlim(XL);
	ylim([0, 2]);
	
	subplot(1, 2, 2);
	hold off;
	plot(sc.Frequencies./1e9, total_cryo, 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
	hold on;	
	Lm1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [lin2dB(cm), lin2dB(cm)], 'LineStyle', '--', 'Color', [0, 0, 0], 'LineWidth', mlw);
	Ls1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], lin2dB([cs_l, cs_l]), 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	Ls2 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], lin2dB([cs_h, cs_h]), 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	grid on;
	xlabel("Frequency (GHz)");
	ylabel("S_{11} + S_{21} (dB)");
	title("Statistical Significance, Logarithmic Scale");
	legend("S_{11} + S_{21}", "Mean", "Standard Deviation");
	ylim([-10, 10]);
	xlim(XL);
	
	barprint("X-LIMIT = " + num2str(XL));
	displ("MEAN = ", mean(cm_lin), " (Lin.), = ", lin2dB(cm_lin), " dB");
	displ("STDEV = ", mean(cs_lin), " (Lin.), = ", lin2dB(cs_lin), " dB");
else
	figure(FIG3);
	subplot(1, 1, 1);
	hold off;
	plot(sc.Frequencies./1e9, dB2lin(total_cryo), 'LineStyle', ':', 'Marker', '.', 'Color', [0.12, 0.7, 0.12]);
	hold on;	
	Lm1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cm, cm], 'LineStyle', '--', 'Color', [0, 0, 0], 'LineWidth', mlw);
	Ls1 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_l, cs_l], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	Ls2 = line([min(sc.Frequencies./1e9), max(sc.Frequencies./1e9)], [cs_h, cs_h], 'LineStyle', ':', 'Color', [0.2, 0.2, 0.2], 'LineWidth', mlw);
	grid on;
	xlabel("Frequency (GHz)");
	ylabel("S_{11} + S_{21}, Linear");
	title("Statistical Significance, Linear Scale");
	legend("S_{11} + S_{21}", "Mean", "Standard Deviation");
	ylim([-10, 10]);
	xlim(XL);
	
	barprint("X-LIMIT = " + num2str(XL));
	displ("MEAN = ", mean(cm_lin), " (Lin.), = ", lin2dB(cm_lin), " dB");
	displ("STDEV = ", mean(cs_lin), " (Lin.), = ", lin2dB(cs_lin), " dB");
end







