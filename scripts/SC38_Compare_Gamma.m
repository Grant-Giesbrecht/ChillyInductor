%% Specify Data

FIG_NUM = 1;
FIG_HOLD = false;

LIMIT_FREQ_BAND = true; % ONly plots frequencies within the band where the datasets overlap

DATA_PATH1 = string(fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps'));
DATA_PATH2 = string(fullfile('/','Volumes','NO NAME', 'NIST September data'));
DATA_PATH3 = string(fullfile('/', 'Volumes', 'NO NAME', 'NIST September data'));

% Paths to datafiles (Target-1 peak)
files = ["gamma_10GHz_500MHzBW_20Oct2023.mat", "gamma_9,87GHz_Target1_24Oct2023.mat", "gamma_9,87GHz_Target1_26Oct2023_r1.mat", "gamma_9,87GHz_Target1_26Oct2023_r2.mat", "gamma_9,87GHz_Target1_26Oct2023_r3.mat", "gamma_9,87GHz_Target1_26Oct2023_r4.mat", "gamma_9,87GHz_Target1_26Oct2023_r5.mat", "gamma_9,87GHz_Target1_26Oct2023_r6.mat", "gamma_9,87GHz_Target1_26Oct2023_r7.mat", "gamma_9,87GHz_Target1_26Oct2023_r8.mat", "gamma_9,87GHz_Target1_26Oct2023_r9.mat"];
dpaths = repmat([DATA_PATH3], 1, numel(files));
normals = repmat([.001], 1, numel(files));


% % Paths to datafiles (9.6-9.7 GHz)
% dpaths = [DATA_PATH3, DATA_PATH3];
% files = ["gamma_9,7GHz_200MHzBW_19Oct2023.mat", "gamma_9,7GHz_200MHzBW_v3,1_18Oct2023.mat"];
% normals = [.001, .001];

%% Detect if reload is required
RELOAD_DATA = ~exist('ALL_DATA', 'var');
if ~RELOAD_DATA
	
	% Check that appropriate data has been loaded
	for idx = 1:numel(dpaths)
		
		% Check that each entry matches requested datafile
		try
			if ~strcmp(ALL_DATA(idx).filename, files(idx))
				RELOAD_DATA = true;
				displ(" --> ALL_DATA contains data for old recipe; Reloading data.");
				break;
			end
		catch
			displ(" --> ALL_DATA contains insufficient points; Reloading data.");
			RELOAD_DATA = true;
			break;
		end
		
	end
end

%% Create Master Data List

if RELOAD_DATA
	
	any_powers = []; % List of all power levels present in any dataset
	univ_powers = []; % List of power levels present in all datasets
	
	disp(" --> Reloading ALL_DATA.");
	
	clear ALL_DATA;

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
		
		% Add to ALL_DATA
		ALL_DATA(idx) = entry;
		
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
	disp(" --> ALL_DATA up-to-date; Skipping reload.");
end

% Pick frequencies to plot
overlap_region = [];
master_region = [];
if LIMIT_FREQ_BAND
	
	disp(" --> Selecting limited frequency band.");
	
	% Scan over each dataset - get region of universal overlap
	for dsidx = 1:numel(ALL_DATA)
		% Initialize
		if dsidx == 1
			overlap_region = [min(ALL_DATA(dsidx).ds.configuration.frequency), max(ALL_DATA(dsidx).ds.configuration.frequency)];
		else
			
			overlap_region(1) = max([overlap_region(1), min(ALL_DATA(dsidx).ds.configuration.frequency)]);
			overlap_region(2) = min([overlap_region(2), max(ALL_DATA(dsidx).ds.configuration.frequency)]);
			
		end
	end
	
	displ("   --> Found overlap region: [", overlap_region(1)./1e9, " GHz - ", overlap_region(2)./1e9, " GHz].");
	
	% Allow one point outside universal region on each end (high/low) for
	% each dataset.
	for dsidx = 1:numel(ALL_DATA)
		
		ds_f = ALL_DATA(dsidx).ds.configuration.frequency;
		
		% Find region within overlap area
		I_match = (ds_f >= overlap_region(1) & ds_f <= overlap_region(2));
		
		% Pad accepted indecies by 1 on each side (lower)
		idx0 = find(I_match, 1, 'first');
		if idx0 > 1 % Add 1 more element if possible
			idx0 = idx0 - 1;
		end
		
		% Pad accepted indecies by 1 on each side (upper)
		idx1 = find(I_match, 1, 'last');
		if idx1 < numel(I_match) % Add 1 more element if possible
			idx1 = idx1 + 1;
		end
		
		% Update entry
		ALL_DATA(dsidx).freq_idx = [idx0, idx1];
		
		% Generate dataset mask
		ds_frequency_points = [ALL_DATA(dsidx).ds.dataset.SG_freq_Hz];
		ALL_DATA(dsidx).mask = (ds_frequency_points >= ds_f(idx0)) & (ds_frequency_points <= ds_f(idx1));
		
		% Copy dataset and mask its values
		ALL_DATA(dsidx).ds_abbrev = ALL_DATA(dsidx).ds;
		ALL_DATA(dsidx).ds_abbrev.dataset = ALL_DATA(dsidx).ds_abbrev.dataset(ALL_DATA(dsidx).mask);
		
		displ("     --> Dataset ", dsidx ,", set frequency indecies: [", ALL_DATA(dsidx).freq_idx(1), " - ", ALL_DATA(dsidx).freq_idx(2), "].");
		
	end
	
end

%% Generate plots

disp(" --> Generating plots.");

powers_plot = univ_powers;

COLORS = [0, 0, 0.6;  0.6, 0, 0; 0, 0.6, 0; 0.7, 0, 0.3];
lw = 1.3;
MKZS = [10, 10, 10, 10, 10];
MARKERS = ['o', '+', '*', '.', 'v'];
% Get number of rows and columns
num_plots = numel(powers_plot);
cols = ceil(sqrt(num_plots));
rows = ceil(num_plots/cols);

% Prepare graph
figure(FIG_NUM);
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
	
	% Plot each dataset
	CM = resamplecmap('parula', numel(ALL_DATA));
% 	CM = COLORS;
	for dsidx = 1:numel(ALL_DATA)
		
		displ("     --> Dataset ", dsidx, " of ", numel(ALL_DATA), ".");
		
		entry = ALL_DATA(dsidx);
		c.Vnorm = entry.NORMAL_VOLTAGE;
		
		% Get selected frequencies for dataset
		if LIMIT_FREQ_BAND
			freqs_band = entry.ds.configuration.frequency(entry.freq_idx(1):entry.freq_idx(2));
		else
			freqs_band = entry.ds.configuration.frequency;
		end
		
% 		displ('freqs_band: ', freqs_band);
		
		% Get CE vs frequency
		CE2 = zeros(1, numel(freqs_band));
		for fidx = 1:numel(freqs_band)
			
			% Update conditions
			c.f0 = freqs_band(fidx);
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(entry.ds_abbrev, c, false);
			
			% Calculate conversion efficiency
			CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			[CE2(fidx), mi2] = max(CE2_);
		end
		
		% Plot data
		dsidx_mod = mod(dsidx, numel(MARKERS));
		if dsidx_mod == 0
			dsidx_mod = numel(MARKERS);
		end
		plot(freqs_band./1e9, CE2, 'LineStyle', ':', 'Marker', MARKERS(dsidx_mod), 'Color', CM(dsidx, :), 'DisplayName', strrep(entry.filename, '_', '\_'), 'LineWidth', lw, 'MarkerSize', MKZS(dsidx_mod));
		hold on;
	end
	
	% Apply subplot settings
	xlabel("Frequency (GHz)");
	ylabel("Conversion Efficiency (%)");
	title("P_{RF} = "+num2str(power) + " dBm");
	grid on;
	legend('Location', 'best');
	
end











































