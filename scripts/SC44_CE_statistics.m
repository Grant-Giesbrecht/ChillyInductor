%% Description
% SC44: Built off of SC38_compare_gammas.m to provide basic data plots
% (like in SC30), but with error bars shown.
%
% Expects all datasets to have same conditions and frequecny points unlike
% SC38.

%% Specify Data

FIG_NUMA = 13; % CE vs freq w/ error bars
FIG_NUMA_h3 = 15; % CE vs freq w/ error bars, 3rd harm CE
FIG_NUMB = 14; % Power vs bias w/ error bars
FIG_NUMB2 = 12; % Power vs bias - 2nd harmonic only
FIG_NUMB3 = 11; % Power vs bias - repeated in triplicate
FIG_HOLD = false;

% Conditions for Plot B: power vs bias w/ error bars
FREQ_PLOT = 9.87e9; % Hz
FREQ_PLOT_T2 = 9.86e9; % Hz - 2nd frequency used in triplicate graphs
FREQ_PLOT_T3 = 9.88e9; % Hz - 3rd frequency used in triplicate graphs
PWR_PLOT = 6; % dBm

LIMIT_FREQ_BAND = true; % ONly plots frequencies within the band where the datasets overlap

DATA_PATH1 = string(fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps'));
DATA_PATH2 = string(fullfile('/','Volumes','NO NAME', 'NIST September data'));
DATA_PATH3 = string(fullfile('/', 'Volumes', 'M6 T7S', 'ARC0 PhD Data', 'RP-21 Kinetic Inductance 2023', 'Data',...
	'group4_extflash', 'NIST September data'));
DATA_PATH4 = string(fullfile('/', 'Volumes', 'M5 PERSONAL', 'CalTest'));

% Paths to datafiles (Target-1 peak)
files = ["gamma_9,87GHz_Target1_24Oct2023.mat", "gamma_9,87GHz_Target1_26Oct2023_r1.mat",...
	"gamma_9,87GHz_Target1_26Oct2023_r2.mat", "gamma_9,87GHz_Target1_26Oct2023_r3.mat", "gamma_9,87GHz_Target1_26Oct2023_r4.mat",...
	"gamma_9,87GHz_Target1_26Oct2023_r5.mat", "gamma_9,87GHz_Target1_26Oct2023_r6.mat", "gamma_9,87GHz_Target1_26Oct2023_r7.mat",...
	"gamma_9,87GHz_Target1_26Oct2023_r8.mat", "gamma_9,87GHz_Target1_26Oct2023_r9.mat"];
% files = ["gamma_9,87GHz_Target1_AusfB_R1_18Nov2023.mat", "gamma_9,87GHz_Target1_AusfB_R2_18Nov2023.mat",...
% 	"gamma_9,87GHz_Target1_AusfB_R3_18Nov2023.mat", "gamma_9,87GHz_Target1_AusfB_R4_18Nov2023.mat",...
% 	"gamma_9,87GHz_Target1_AusfB_R5_18Nov2023.mat", "gamma_9,87GHz_Target1_AusfB_R6_18Nov2023.mat",...
% 	"gamma_9,87GHz_Target1_AusfB_R7_18Nov2023.mat", "gamma_9,87GHz_Target1_AusfB_R8_18Nov2023.mat",...
% 	"gamma_9,87GHz_Target1_AusfB_R9_18Nov2023.mat", "gamma_9,87GHz_Target1_AusfB_R10_18Nov2023.mat"];
dpaths = repmat(string(DATA_PATH3), 1, numel(files));
normals = repmat([.001], 1, numel(files));

%% Detect if reload is required
RELOAD_DATA = ~exist('ALL_DATA44', 'var');
if ~RELOAD_DATA
	
	% Check that appropriate data has been loaded
	for idx = 1:numel(dpaths)
		
		% Check that each entry matches requested datafile
		try
			if ~strcmp(ALL_DATA44(idx).filename, files(idx))
				RELOAD_DATA = true;
				displ(" --> ALL_DATA44 contains data for old recipe; Reloading data.");
				break;
			end
		catch
			displ(" --> ALL_DATA44 contains insufficient points; Reloading data.");
			RELOAD_DATA = true;
			break;
		end
		
	end
end

%% Create Master Data List

if RELOAD_DATA
	
	any_powers = []; % List of all power levels present in any dataset
	univ_powers = []; % List of power levels present in all datasets
	
	disp(" --> Reloading ALL_DATA44.");
	
	clear ALL_DATA44;

	% Load all files
	for idx = numel(dpaths):-1:1
		
		% Create struct
		entry = struct('filename', files(idx) );
		entry.NORMAL_VOLTAGE = normals(idx);
		entry.dpath = dpaths(idx);
		
		% Laod file
		warning off;
		load(fullfile(entry.dpath, entry.filename));
		warning on;
		entry.ds = ds;
		entry.freq_idx = [-1, -1]; % Indecies of frequencies to plot
		entry.mask = []; % Mask s.t. only includes correct frequencies
		
		% Add to ALL_DATA44
		ALL_DATA44(idx) = entry;
		
		% Update any_powers
		any_powers = unique([any_powers, entry.ds.configuration.RF_power]);
		
		% Update univ_powers
		if idx == numel(dpaths)
			univ_powers = entry.ds.configuration.RF_power;
		else
			univ_powers = intersect(univ_powers, entry.ds.configuration.RF_power);
		end
		
		displ("   --> Loaded file ", numel(dpaths)+1-idx, " of ", numel(dpaths), ".");
	end

	clear ds;
else
	disp(" --> ALL_DATA44 up-to-date; Skipping reload.");
end

% Pick frequencies to plot
overlap_region = [];
master_region = [];
if LIMIT_FREQ_BAND
	
	disp(" --> Selecting limited frequency band.");
	
	% Scan over each dataset - get region of universal overlap
	for dsidx = 1:numel(ALL_DATA44)
		% Initialize
		if dsidx == 1
			overlap_region = [min(ALL_DATA44(dsidx).ds.configuration.frequency), max(ALL_DATA44(dsidx).ds.configuration.frequency)];
		else
			
			overlap_region(1) = max([overlap_region(1), min(ALL_DATA44(dsidx).ds.configuration.frequency)]);
			overlap_region(2) = min([overlap_region(2), max(ALL_DATA44(dsidx).ds.configuration.frequency)]);
			
		end
	end
	
	displ("   --> Found overlap region: [", overlap_region(1)./1e9, " GHz - ", overlap_region(2)./1e9, " GHz].");
	
	% Allow one point outside universal region on each end (high/low) for
	% each dataset.
	for dsidx = 1:numel(ALL_DATA44)
		
		ds_f = ALL_DATA44(dsidx).ds.configuration.frequency;
		
		% Find region within overlap area
		I_match = (ds_f >= overlap_region(1) & ds_f <= overlap_region(2));
		
		% Pad accepted indecies by 1 on each side (lower)
		idx0 = find(I_match, 1, 'first');
		
		% Pad accepted indecies by 1 on each side (upper)
		idx1 = find(I_match, 1, 'last');
		
		% Update entry
		ALL_DATA44(dsidx).freq_idx = [idx0, idx1];
		
		% Generate dataset mask
		ds_frequency_points = [ALL_DATA44(dsidx).ds.dataset.SG_freq_Hz];
		ALL_DATA44(dsidx).mask = (ds_frequency_points >= ds_f(idx0)) & (ds_frequency_points <= ds_f(idx1));
		
		% Copy dataset and mask its values
		ALL_DATA44(dsidx).ds_abbrev = ALL_DATA44(dsidx).ds;
		ALL_DATA44(dsidx).ds_abbrev.dataset = ALL_DATA44(dsidx).ds_abbrev.dataset(ALL_DATA44(dsidx).mask);
		
		displ("     --> Dataset ", dsidx ,", set frequency indecies: [", ALL_DATA44(dsidx).freq_idx(1), " - ",...
			ALL_DATA44(dsidx).freq_idx(2), "].");
		
	end
	
end

%% Generate plots - CE vs Frequency

disp(" --> Generating plots.");

powers_plot = univ_powers;
powers_plot = 6;

COLORS = [0, 0, 0.6;  0.6, 0, 0; 0, 0.6, 0; 0.7, 0, 0.3];
lw = 1.3;
MKZS = [10, 10, 10, 10, 10];
MARKERS = ['o', '+', '*', '.', 'v'];
% Get number of rows and columns
num_plots = numel(powers_plot);
cols = ceil(sqrt(num_plots));
rows = ceil(num_plots/cols);

% Prepare graph
figure(FIG_NUMA);
if ~FIG_HOLD
	for np = 1:num_plots
		subplot(rows, cols, np);
		hold off;
	end
else
	for np = 1:num_plots
		subplot(rows, cols, np);
		hold on;
	end
end

% Prepare graph
figure(FIG_NUMA_h3);
if ~FIG_HOLD
	for np = 1:num_plots
		subplot(rows, cols, np);
		hold off;
	end
else
	for np = 1:num_plots
		subplot(rows, cols, np);
		hold on;
	end
end

% Generate local conditions struct
c = defaultConditions();
c.SG_power = 0;
c.convert_to_W = 1;
c.Vnorm = 1e-3;



% Plot each power level
for pidx = 1:numel(powers_plot)
	
	displ("   --> Power level ", pidx, " of ", numel(powers_plot), ".");
	
	% Set subplot
	subplot(rows, cols, pidx);
	
	% Power level
	power = powers_plot(pidx);
	c.SG_power = power;
	
	freqs_band = ALL_DATA44(1).ds.configuration.frequency;
	
	% Create meister
	CE2 = zeros(numel(ALL_DATA44), numel(freqs_band));
	CE3 = zeros(numel(ALL_DATA44), numel(freqs_band));
	
	% Analyze each dataset
	for dsidx = 1:numel(ALL_DATA44)
		
		displ("     --> Dataset ", dsidx, " of ", numel(ALL_DATA44), ".");
		
		entry = ALL_DATA44(dsidx);
		c.Vnorm = entry.NORMAL_VOLTAGE;
		
% 		% Get selected frequencies for dataset
% 		if LIMIT_FREQ_BAND
% 			freqs_band = entry.ds.configuration.frequency(entry.freq_idx(1):entry.freq_idx(2));
% 		else
% 			
% 		end
		
		% Scan over each frequency
		for fidx = 1:numel(freqs_band)
			
			
			% Update conditions
			c.f0 = freqs_band(fidx);
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(entry.ds_abbrev, c, false);
			
			% Calculate conversion efficiency
			CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			[CE2(dsidx, fidx), mi2] = max(CE2_);
			
			% Calculate conversion efficiency
			CE3_ = abs(harm_struct.h3)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			[CE3(dsidx, fidx), mi2] = max(CE3_);
		end
		
	end
	
	% Get mean and stdev
	CE2_means = mean(CE2, 1);
	CE2_stds = std(CE2, 1);
	
	% Get mean and stdev
	CE3_means = mean(CE3, 1);
	CE3_stds = std(CE3, 1);
	
	% Plot data
	dsidx_mod = mod(dsidx, numel(MARKERS));
	if dsidx_mod == 0
		dsidx_mod = numel(MARKERS);
	end
	
	scale = 1;
	
	figure(FIG_NUMA);
% 	yyaxis left;
	% [0, 0.3, 0.75]
	plot(freqs_band./1e9, CE2_means, 'LineStyle', '--', 'Color', [0.8, 0, 0], 'DisplayName',...
		strrep(entry.filename, '_', '\_'), 'LineWidth', lw);
	hold on;
% 	yyaxis right;
% 	plot(freqs_band./1e9, CE2_stds);
% 	hold on;
% 	yyaxis left;
	addPlotErrorbars(freqs_band./1e9, CE2_means, CE2_stds.*scale, true, 200);
	
	% Apply subplot settings
	xlabel("Frequency (GHz)");
	ylabel("Conversion Efficiency (%)");
	title("Conversion Efficiency Frequency Response, P_{RF} = "+num2str(power) + " dBm");
	grid on;
% 	legend('Location', 'best');
	
	font_size = 17;
	set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
	
	if scale ~= 1
		text(9.8575, font_size, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
	end
	
	figure(FIG_NUMA_h3);
	plot(freqs_band./1e9, CE3_means, 'LineStyle', '--', 'Color', [0.8, 0, 0], 'DisplayName',...
		strrep(entry.filename, '_', '\_'), 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', 10);
	hold on;
	addPlotErrorbars(freqs_band./1e9, CE3_means, CE3_stds.*scale, false, 200);
	xlabel("Frequency (GHz)");
	ylabel("Conversion Efficiency (%)");
	title("3rd Harmonic Conversion Efficiency Frequency Response, P_{RF} = "+num2str(power) + " dBm");
	grid on;
	set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
	if scale ~= 1
		text(9.8575, 14, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
	end
	figure(FIG_NUMA);
end

%% Generate Plots - Power vs Bias

figure(FIG_NUMB)
hold off;

V_ph = ALL_DATA44(1).ds.configuration.bias_V;
P1H = zeros(numel(ALL_DATA44), numel(V_ph));
P2H = zeros(numel(ALL_DATA44), numel(V_ph));
P3H = zeros(numel(ALL_DATA44), numel(V_ph));

% Scan over all datasets
for dsidx = 1:numel(ALL_DATA44)
	
	% Update conditions
	c.f0 = FREQ_PLOT;
	c.SG_power = PWR_PLOT;
	
	entry = ALL_DATA44(dsidx);
	
	% Extract data
	[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(entry.ds_abbrev, c, false);
	
	% Verify matching bias
	if Vsweep ~= V_ph
		error("Bias points don't match.");
		return;
	end
	
	% Save data
	P1H(dsidx, :) = abs(harm_struct.h1);
	P2H(dsidx, :) = abs(harm_struct.h2);
	P3H(dsidx, :) = abs(harm_struct.h3);
	
end


% Calculate stats
P1H_means = mean(P1H, 1);
P1H_stds = std(P1H, 1);
P2H_means = mean(P2H, 1);
P2H_stds = std(P2H, 1);
P3H_means = mean(P3H, 1);
P3H_stds = std(P3H, 1);

CSZ = 10;
c1 = [0, 0.2, 0.75];
c2 = [0, 0.75, 0.3];
c3 = [0.8, 0, 0];

scale = 1;

P1Hm_dBm = cvrt(P1H_means, 'W', 'dBm');

Inan = isnan(P1Hm_dBm);
P1Hm_dBm = P1Hm_dBm(~Inan);
V_ph = V_ph(~Inan);

P1HeL_dBm = cvrt(P1H_means(~Inan)-P1H_stds(~Inan)*scale, 'W', 'dBm');
P1HeH_dBm = cvrt(P1H_means(~Inan)+P1H_stds(~Inan)*scale, 'W', 'dBm');

P2Hm_dBm = cvrt(P2H_means(~Inan), 'W', 'dBm');
P2HeL_dBm = cvrt(P2H_means(~Inan)-P2H_stds(~Inan)*scale, 'W', 'dBm');
P2HeH_dBm = cvrt(P2H_means(~Inan)+P2H_stds(~Inan)*scale, 'W', 'dBm');

P3Hm_dBm = cvrt(P3H_means(~Inan), 'W', 'dBm');
P3HeL_dBm = cvrt(P3H_means(~Inan)-P3H_stds(~Inan)*scale, 'W', 'dBm');
P3HeH_dBm = cvrt(P3H_means(~Inan)+P3H_stds(~Inan)*scale, 'W', 'dBm');



circs_on = false;
MKZ = 8;

figure(14);
plot(V_ph./105.*1e3, P1Hm_dBm, 'LineStyle', '--', 'Color', c1, 'DisplayName', "Fundamental", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
hold on;
addPlotErrorbars2(V_ph./105.*1e3, P1Hm_dBm, P1HeL_dBm, P1HeH_dBm, circs_on, CSZ, c1);

plot(V_ph./105.*1e3, P2Hm_dBm, 'LineStyle', '--', 'Color', c2, 'DisplayName', "2nd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P2Hm_dBm, P2HeL_dBm, P2HeH_dBm, circs_on, CSZ, c2);

plot(V_ph./105.*1e3, P3Hm_dBm, 'LineStyle', '--', 'Color', c3, 'DisplayName', "3rd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P3Hm_dBm, P3HeL_dBm, P3HeH_dBm, circs_on, CSZ, c3);

grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Harmonic Power at "+num2str(FREQ_PLOT/1e9) + " GHz and " + num2str(PWR_PLOT)+ " dBm P_{RF}");

legend('Location', 'Best');

font_size = 17;
set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
if scale ~=1
	text(.6, -10, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
end

%% Generate Plots - Power vs Bias (Repeat above in triplicate for TAS paper T's suggestion)

figure(FIG_NUMB3)
hold off;

V_ph = ALL_DATA44(1).ds.configuration.bias_V;
P1H_T2 = zeros(numel(ALL_DATA44), numel(V_ph));
P2H_T2 = zeros(numel(ALL_DATA44), numel(V_ph));
P3H_T2 = zeros(numel(ALL_DATA44), numel(V_ph));

P1H_T3 = zeros(numel(ALL_DATA44), numel(V_ph));
P2H_T3 = zeros(numel(ALL_DATA44), numel(V_ph));
P3H_T3 = zeros(numel(ALL_DATA44), numel(V_ph));

% Scan over all datasets
for dsidx = 1:numel(ALL_DATA44)
	
	%-------------------- T2 ----------------
	
	% Update conditions
	c.f0 = FREQ_PLOT_T2;
	c.SG_power = PWR_PLOT;
	
	entry = ALL_DATA44(dsidx);
	
	% Extract data
	[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(entry.ds_abbrev, c, false);
	
	% Verify matching bias
	if Vsweep ~= V_ph
		error("Bias points don't match.");
		return;
	end
	
	% Save data
	P1H_T2(dsidx, :) = abs(harm_struct.h1);
	P2H_T2(dsidx, :) = abs(harm_struct.h2);
	P3H_T2(dsidx, :) = abs(harm_struct.h3);
	
	%-------------------- T3 ----------------
	
	% Update conditions
	c.f0 = FREQ_PLOT_T3;
	c.SG_power = PWR_PLOT;
	
	entry = ALL_DATA44(dsidx);
	
	% Extract data
	[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(entry.ds_abbrev, c, false);
	
	% Verify matching bias
	if Vsweep ~= V_ph
		error("Bias points don't match.");
		return;
	end
	
	% Save data
	P1H_T3(dsidx, :) = abs(harm_struct.h1);
	P2H_T3(dsidx, :) = abs(harm_struct.h2);
	P3H_T3(dsidx, :) = abs(harm_struct.h3);
	
end


% Calculate stats
P1H_T2_means = mean(P1H_T2, 1);
P1H_T2_stds = std(P1H_T2, 1);
P2H_T2_means = mean(P2H_T2, 1);
P2H_T2_stds = std(P2H_T2, 1);
P3H_T2_means = mean(P3H_T2, 1);
P3H_T2_stds = std(P3H_T2, 1);

P1H_T3_means = mean(P1H_T3, 1);
P1H_T3_stds = std(P1H_T3, 1);
P2H_T3_means = mean(P2H_T3, 1);
P2H_T3_stds = std(P2H_T3, 1);
P3H_T3_means = mean(P3H_T3, 1);
P3H_T3_stds = std(P3H_T3, 1);

P1Hm_T2_dBm = cvrt(P1H_T2_means, 'W', 'dBm');
P1Hm_T3_dBm = cvrt(P1H_T3_means, 'W', 'dBm');

Inan = isnan(P1Hm_T2_dBm);
P1Hm_T2_dBm = P1Hm_T2_dBm(~Inan);
P1Hm_T3_dBm = P1Hm_T3_dBm(~Inan);
V_ph = V_ph(~Inan);

P1HeL_T2_dBm = cvrt(P1H_T2_means(~Inan)-P1H_T2_stds(~Inan)*scale, 'W', 'dBm');
P1HeH_T2_dBm = cvrt(P1H_T2_means(~Inan)+P1H_T2_stds(~Inan)*scale, 'W', 'dBm');

P2Hm_T2_dBm = cvrt(P2H_T2_means(~Inan), 'W', 'dBm');
P2HeL_T2_dBm = cvrt(P2H_T2_means(~Inan)-P2H_T2_stds(~Inan)*scale, 'W', 'dBm');
P2HeH_T2_dBm = cvrt(P2H_T2_means(~Inan)+P2H_T2_stds(~Inan)*scale, 'W', 'dBm');

P3Hm_T2_dBm = cvrt(P3H_T2_means(~Inan), 'W', 'dBm');
P3HeL_T2_dBm = cvrt(P3H_T2_means(~Inan)-P3H_T2_stds(~Inan)*scale, 'W', 'dBm');
P3HeH_T2_dBm = cvrt(P3H_T2_means(~Inan)+P3H_T2_stds(~Inan)*scale, 'W', 'dBm');

P1HeL_T3_dBm = cvrt(P1H_T3_means(~Inan)-P1H_T3_stds(~Inan)*scale, 'W', 'dBm');
P1HeH_T3_dBm = cvrt(P1H_T3_means(~Inan)+P1H_T3_stds(~Inan)*scale, 'W', 'dBm');

P2Hm_T3_dBm = cvrt(P2H_T3_means(~Inan), 'W', 'dBm');
P2HeL_T3_dBm = cvrt(P2H_T3_means(~Inan)-P2H_T3_stds(~Inan)*scale, 'W', 'dBm');
P2HeH_T3_dBm = cvrt(P2H_T3_means(~Inan)+P2H_T3_stds(~Inan)*scale, 'W', 'dBm');

P3Hm_T3_dBm = cvrt(P3H_T3_means(~Inan), 'W', 'dBm');
P3HeL_T3_dBm = cvrt(P3H_T3_means(~Inan)-P3H_T3_stds(~Inan)*scale, 'W', 'dBm');
P3HeH_T3_dBm = cvrt(P3H_T3_means(~Inan)+P3H_T3_stds(~Inan)*scale, 'W', 'dBm');



circs_on = false;
MKZ = 8;

%----------- Copy 1----------------

figure(FIG_NUMB3);
subplot(1, 3, 2);
hold off;
plot(V_ph./105.*1e3, P1Hm_dBm, 'LineStyle', '--', 'Color', c1, 'DisplayName', "Fundamental", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
hold on;
addPlotErrorbars2(V_ph./105.*1e3, P1Hm_dBm, P1HeL_dBm, P1HeH_dBm, circs_on, CSZ, c1);

plot(V_ph./105.*1e3, P2Hm_dBm, 'LineStyle', '--', 'Color', c2, 'DisplayName', "2nd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P2Hm_dBm, P2HeL_dBm, P2HeH_dBm, circs_on, CSZ, c2);

plot(V_ph./105.*1e3, P3Hm_dBm, 'LineStyle', '--', 'Color', c3, 'DisplayName', "3rd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P3Hm_dBm, P3HeL_dBm, P3HeH_dBm, circs_on, CSZ, c3);

grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Harmonic Power at "+num2str(FREQ_PLOT/1e9) + " GHz and " + num2str(PWR_PLOT)+ " dBm P_{RF}");
legend('Location', 'Best');

font_size = 17;
set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
if scale ~=1
	text(.6, -10, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
end
%----------- Copy 2 ----------------

subplot(1, 3, 1);
hold off;
plot(V_ph./105.*1e3, P1Hm_T2_dBm, 'LineStyle', '--', 'Color', c1, 'DisplayName', "Fundamental", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
hold on;
addPlotErrorbars2(V_ph./105.*1e3, P1Hm_T2_dBm, P1HeL_T2_dBm, P1HeH_T2_dBm, circs_on, CSZ, c1);

plot(V_ph./105.*1e3, P2Hm_T2_dBm, 'LineStyle', '--', 'Color', c2, 'DisplayName', "2nd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P2Hm_T2_dBm, P2HeL_T2_dBm, P2HeH_T2_dBm, circs_on, CSZ, c2);

plot(V_ph./105.*1e3, P3Hm_T2_dBm, 'LineStyle', '--', 'Color', c3, 'DisplayName', "3rd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P3Hm_T2_dBm, P3HeL_T2_dBm, P3HeH_T2_dBm, circs_on, CSZ, c3);

grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Harmonic Power at "+num2str(FREQ_PLOT_T2/1e9) + " GHz and " + num2str(PWR_PLOT)+ " dBm P_{RF}");
legend('Location', 'Best');

font_size = 17;
set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
if scale ~=1
	text(.6, -10, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
end
%----------- Copy 3 ----------------

subplot(1, 3, 3);
hold off;
plot(V_ph./105.*1e3, P1Hm_T3_dBm, 'LineStyle', '--', 'Color', c1, 'DisplayName', "Fundamental", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
hold on;
addPlotErrorbars2(V_ph./105.*1e3, P1Hm_T3_dBm, P1HeL_T3_dBm, P1HeH_T3_dBm, circs_on, CSZ, c1);

plot(V_ph./105.*1e3, P2Hm_T3_dBm, 'LineStyle', '--', 'Color', c2, 'DisplayName', "2nd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P2Hm_T3_dBm, P2HeL_T3_dBm, P2HeH_T3_dBm, circs_on, CSZ, c2);

plot(V_ph./105.*1e3, P3Hm_T3_dBm, 'LineStyle', '--', 'Color', c3, 'DisplayName', "3rd Harmonic", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
addPlotErrorbars2(V_ph./105.*1e3, P3Hm_T3_dBm, P3HeL_T3_dBm, P3HeH_T3_dBm, circs_on, CSZ, c3);

grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Harmonic Power at "+num2str(FREQ_PLOT_T3/1e9) + " GHz and " + num2str(PWR_PLOT)+ " dBm P_{RF}");
legend('Location', 'Best');

font_size = 17;
set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
if scale ~=1
	text(.6, -10, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
end

%% Read and plot ADS Data

% Path to ADS data
ads_path = '/Volumes/M6 T7S/ARC0 PhD Data/RP-21 Kinetic Inductance 2023/Data/group4_extflash/ADS Sim Data/';



% Load data
[harm_idx, freq_SG, Vbias, Pload_mag, Pload_deg] = load_ADS_2D(fullfile(ads_path, "Freq_DC_Sweep_100K_ForPub.csv"));

% ----- Plot, at single frequency, harmonic powers versus freqeuncy ------

% FREQ_PLOT = 9.87e9; % Hz
% PWR_PLOT = 6; % dBm

% CONSTANT_LOSS1 = 8.7; % Constant loss to apply to simulated values, dB <- mean of top and bottom
% CONSTANT_LOSS1 = 11.7; % Constant loss to apply to simulated values, dB  <- normalized
CONSTANT_LOSS1 = 10.9; % Constant loss to apply to simulated values, dB <- lower limit of osc

CONSTANT_LOSS2 = 15.3; % Constant loss to apply to simulated values, dB

CONSTANT_LOSS3 = 24.5; % Constant loss to apply to simulated values, dB
% CONSTANT_LOSS3 = 28.5; % Constant loss to apply to simulated values, dB
% CONSTANT_LOSS3 = 29.4; % Constant loss to apply to simulated values, dB

max_allowed_bias = 1.15;

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
V_h1 = zeros(1, numel(biases));
V_h2 = zeros(1, numel(biases));
V_h3 = zeros(1, numel(biases));

FREQ_PLOT = 9.88e9;

% Create frequency mask
Iw = (freq_SG == FREQ_PLOT./1e9);

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
	V_h1(idx_b) = p1;

	p2 = Pload_mag(submask & Ih2);
	V_h2(idx_b) = p2;

	p3 = Pload_mag(submask & Ih3);
	V_h3(idx_b) = p3;

end


Z0 = 50;
P_h1 = cvrt(V_h1.^2./Z0, 'W', 'dBm');
P_h2 = cvrt(V_h2.^2./Z0, 'W', 'dBm');
P_h3 = cvrt(V_h3.^2./Z0, 'W', 'dBm');

c1s = [0, 0.1, 0.65];
c2s = [0, 0.65, 0.15];
c3s = [0.6, 0, 0];

LSs = ':';
MKs = '+';

SIM_RES = 50;

figure(FIG_NUMB);
plot(biases./SIM_RES.*1e3, P_h1-CONSTANT_LOSS1, 'LineStyle', LSs, 'Color', c1s, 'DisplayName', "Fundamental (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h2-CONSTANT_LOSS2, 'LineStyle', LSs, 'Color', c2s, 'DisplayName', "2nd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h3-CONSTANT_LOSS3, 'LineStyle', LSs, 'Color', c3s, 'DisplayName', "3rd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
xlim([0, V_ph(end)./105.*1e3]);

figure(FIG_NUMB3);
subplot(1, 3, 2);
plot(biases./SIM_RES.*1e3, P_h1-CONSTANT_LOSS1, 'LineStyle', LSs, 'Color', c1s, 'DisplayName', "Fundamental (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h2-CONSTANT_LOSS2, 'LineStyle', LSs, 'Color', c2s, 'DisplayName', "2nd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h3-CONSTANT_LOSS3, 'LineStyle', LSs, 'Color', c3s, 'DisplayName', "3rd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
xlim([0, V_ph(end)./105.*1e3]);

P_h2_B2 = P_h2;

% ----- Plot,over frequency, max CE at each freq ------

CE2maxes = zeros(1, numel(freqs));

idx_f = 0;
for f = freqs
	idx_f = idx_f + 1;
	
	% Create frequency mask
	Iw = (freq_SG == f);
	
	% Create blank array for data
	CE2_sub = zeros(1, numel(biases));
	
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
		V_h1 = p1;
		
		p2 = Pload_mag(submask & Ih2);
		V_h2 = p2;
		
		p3 = Pload_mag(submask & Ih3);
		V_h3 = p3;
		
		Z0 = 50;
		P_h1 = V_h1.^2./Z0;
		P_h2 = V_h2.^2./Z0;
		P_h3 = V_h3.^2./Z0;
		
		CE2_sub(idx_b) = 100.*P_h2./(P_h1 + P_h2 + P_h3);
	end
	
	% Save max to master array
	CE2maxes(idx_f) = max(CE2_sub);
	
end

% Add to figure
figure(FIG_NUMA);
plot(freqs, CE2maxes, 'LineStyle', LSs, 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
legend("Measured", "Simulated");

%% Read and plot ADS Data (Repeat ADS analysis in triplicate)

% Create grid
[B, F] = meshgrid(biases, freqs);

% Create harmonic masks
Ih1 = (harm_idx == 1);
Ih2 = (harm_idx == 2);
Ih3 = (harm_idx == 3);

% Create power variable
V_h1_T2 = zeros(1, numel(biases));
V_h2_T2 = zeros(1, numel(biases));
V_h3_T2 = zeros(1, numel(biases));

V_h1_T3 = zeros(1, numel(biases));
V_h2_T3 = zeros(1, numel(biases));
V_h3_T3 = zeros(1, numel(biases));

% Create frequency mask
Iw_T2 = (freq_SG == FREQ_PLOT_T2./1e9);
Iw_T3 = (freq_SG == FREQ_PLOT_T3./1e9);

idx_b = 0;
for b = biases
	idx_b = idx_b + 1;

	% Create bias and sub masks
	Ib = (Vbias == b);
	submask_T2 = Iw_T2 & Ib;
	submask_T3 = Iw_T3 & Ib;
	
	% ------------ T2 -------------
	
	% Get fundamental
	p1 = Pload_mag(submask_T2 & Ih1);
	if numel(p1) ~= 1
		warning("Wrong number of points");
	end
	V_h1_T2(idx_b) = p1;
	
	p2 = Pload_mag(submask_T2 & Ih2);
	V_h2_T2(idx_b) = p2;
	
	p3 = Pload_mag(submask_T2 & Ih3);
	V_h3_T2(idx_b) = p3;
	
	% ------------ T2 -------------
	
	% Get fundamental
	p1 = Pload_mag(submask_T3 & Ih1);
	if numel(p1) ~= 1
		warning("Wrong number of points");
	end
	V_h1_T3(idx_b) = p1;
	
	p2 = Pload_mag(submask_T3 & Ih2);
	V_h2_T3(idx_b) = p2;
	
	p3 = Pload_mag(submask_T3 & Ih3);
	V_h3_T3(idx_b) = p3;
	
end


Z0 = 50;
P_h1_T2 = cvrt(V_h1_T2.^2./Z0, 'W', 'dBm');
P_h2_T2 = cvrt(V_h2_T2.^2./Z0, 'W', 'dBm');
P_h3_T2 = cvrt(V_h3_T2.^2./Z0, 'W', 'dBm');

P_h1_T3 = cvrt(V_h1_T3.^2./Z0, 'W', 'dBm');
P_h2_T3 = cvrt(V_h2_T3.^2./Z0, 'W', 'dBm');
P_h3_T3 = cvrt(V_h3_T3.^2./Z0, 'W', 'dBm');

figure(FIG_NUMB3);
subplot(1, 3, 1);

plot(biases./SIM_RES.*1e3, P_h1_T2-CONSTANT_LOSS1, 'LineStyle', LSs, 'Color', c1s, 'DisplayName', "Fundamental (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h2_T2-CONSTANT_LOSS2, 'LineStyle', LSs, 'Color', c2s, 'DisplayName', "2nd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h3_T2-CONSTANT_LOSS3, 'LineStyle', LSs, 'Color', c3s, 'DisplayName', "3rd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
xlim([0, V_ph(end)./105.*1e3]);

subplot(1, 3, 3);

plot(biases./SIM_RES.*1e3, P_h1_T3-CONSTANT_LOSS1, 'LineStyle', LSs, 'Color', c1s, 'DisplayName', "Fundamental (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h2_T3-CONSTANT_LOSS2, 'LineStyle', LSs, 'Color', c2s, 'DisplayName', "2nd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
plot(biases./SIM_RES.*1e3, P_h3_T3-CONSTANT_LOSS3, 'LineStyle', LSs, 'Color', c3s, 'DisplayName', "3rd Harmonic (Sim)", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
xlim([0, V_ph(end)./105.*1e3]);

%% Generate Plots - Power vs Bias (But only for 2nd harmonic : for IMS paper)

figure(FIG_NUMB2)
hold off;

circs_on = false;
MKZ = 8;
plot(V_ph./105.*1e3, P2Hm_dBm, 'LineStyle', '--', 'Color', c1, 'DisplayName', "Measurement", 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', MKZ);
figure(FIG_NUMB2);
hold on;
addPlotErrorbars2(V_ph./105.*1e3, P2Hm_dBm, P2HeL_dBm, P2HeH_dBm, circs_on, CSZ, c1);
plot(biases./SIM_RES.*1e3, P_h2_B2-CONSTANT_LOSS2, 'LineStyle', LSs, 'Color', c1s, 'DisplayName', "Simulation", 'LineWidth', lw, 'Marker', MKs, 'MarkerSize', MKZ);
xlim([0, V_ph(end)./105.*1e3]);

grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("Second Harmonic Power at f_0 = "+num2str(FREQ_PLOT/1e9) + " GHz, P_{RF} = " + num2str(PWR_PLOT)+ " dBm");
pbaspect([ 1.0000    0.3652    0.3652]);

legend('Location', 'SouthEast');

font_size = 17;
set(gca,'FontSize', font_size, 'FontName', 'Times New Roman');
if scale ~=1
	text(.6, -10, "Error bars scaled by "+num2str(scale)+" for visibility.", 'FontSize', font_size, 'FontName', 'Times New Roman');
end



%% Define functions



function addPlotErrorbars(Xdata, Ydata, Yerror, add_circ, circ_scaling, bar_color)
	
	if ~exist('bar_color', 'var')
		bar_color = [0.8, 0, 0];
	end
	
	% Scan over each point
	for idx = 1:numel(Yerror)
		line([Xdata(idx), Xdata(idx)], [Ydata(idx)-Yerror(idx), Ydata(idx)+Yerror(idx)], 'LineStyle', '-', 'Color', bar_color, 'LineWidth', 1.5, 'Marker', '_' , 'HandleVisibility', 'off');		
	end
	
	if add_circ
		scatter(Xdata, Ydata, Yerror.*circ_scaling, 'LineWidth', 1.5, 'MarkerEdgeColor', [0.8, 0, 0] , 'HandleVisibility', 'off');
	end
end

function addPlotErrorbars2(Xdata, Ydata, YerrorL, YerrorH, add_circ, circ_scaling, bar_color)
	
	if ~exist('bar_color', 'var')
		bar_color = [0.8, 0, 0];
	end
	
	% Scan over each point
	for idx = 1:numel(YerrorL)
		line([Xdata(idx), Xdata(idx)], [YerrorL(idx), YerrorH(idx)], 'LineStyle', '-', 'Color', bar_color, 'LineWidth', 1.5, 'Marker', '_' , 'HandleVisibility', 'off');		
	end
	
	if add_circ
		scatter(Xdata, Ydata, abs(YerrorL+YerrorH)/2.*circ_scaling, 'LineWidth', 1.5, 'MarkerEdgeColor', bar_color , 'HandleVisibility', 'off');
	end
end