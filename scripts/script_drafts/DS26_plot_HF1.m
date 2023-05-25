%% _Plot_HF1
%
% Previously named: DS26
%
%

%% CONTROL
% Discrete sections at bottom allow alteration of data zoom.

%% Set Conditions

split_freqs = [9.7e9];

%% Plot Measured Data

% Import data
load(dataset_path("DS8_2BandHarmSweep_HF1.mat"));

% Read parameters from data and save to local variables
SG_pwr = ld(1).SG_power_dBm;
all_harms = [ld.harmonic];
harms = unique(all_harms);
num_harms = numel(harms);
all_freq = [ld.SG_freq_Hz];
freqs = unique(all_freq);

% Create data arrays
S21 = zeros(1, numel(ld));
avg_S21 = zeros(1, numel(freqs)*num_harms);
std_S21 = zeros(1, numel(freqs)*num_harms);
harm_list = zeros(1, numel(freqs)*num_harms);
freq_list = zeros(1, numel(freqs)*num_harms);

% Get conversion parameters
a2 = sqrt(cvrt(-10, 'dBm', 'W'));
a_SG = sqrt(cvrt(SG_pwr, 'dBm', 'W'));

% Get mean of VNA data for b1a2
for idx = 1:numel(S21)
	
	% Get mean of data (each point in ld array contains a whole sweep)
	b1a2 = mean(ld(idx).VNA_data.data(1,:));

	% Convert a1b2 to S21
	S21(idx) = abs(b1a2).*a2./a_SG;
	
end

% Average by frequency & harmonic
idx = 0;
for f = freqs
	for h = harms
		idx = idx + 1;

		% Get mask
		I = (f == all_freq) & (all_harms == h);
		
		% Save results
		avg_S21(idx) = mean(S21(I));
		std_S21(idx) = std(S21(I));
		harm_list(idx) = h;
		freq_list(idx) = f;
	end
end

% Get V and I
V = avg_S21.*a_SG.*sqrt(50);
Iac = V./50.*sqrt(2);

base_tick = 0.05;
fb1 = [9.1, 9.5];
fb1_h1 = (freq_list < split_freqs(1)) & (harm_list == 1);
fb1_h2 = (freq_list < split_freqs(1)) & (harm_list == 2);
fb1_h3 = (freq_list < split_freqs(1)) & (harm_list == 3);

fb2 = [9.8, 10.2];
fb2_h1 = (freq_list >= split_freqs(1)) & (harm_list == 1);
fb2_h2 = (freq_list >= split_freqs(1)) & (harm_list == 2);
fb2_h3 = (freq_list >= split_freqs(1)) & (harm_list == 3);

c4 = [202, 129, 1]./255; % Brightestst orange in the circle aroudn the blue circle (c3), both in top, of 208897-20.
c1 = c4; %[255, 242, 29]./255; % Bright yellow point in 'screen-used-lcars'

lw = 1;
mksz = 10;

figure(1);
subplot(3, 1, 1);
hold off;
plot(freq_list(fb1_h1)./1e9, Iac(fb1_h1).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 1: Fundamental")
force0y;
xlim(fb1);
setxtick(base_tick, false);

subplot(3, 1, 2);
hold off;
plot(freq_list(fb1_h2)./1e9.*2, Iac(fb1_h2).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 1: 2nd Harmonic")
force0y;
xlim(fb1.*2);
setxtick(base_tick*2, false);

subplot(3, 1, 3);
hold off;
plot(freq_list(fb1_h3)./1e9.*3, Iac(fb1_h3).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 1: 3rd Harmonic")
force0y;
xlim(fb1.*3);
setxtick(base_tick*3, false);

figure(2);
subplot(3, 1, 1);
hold off;
plot(freq_list(fb2_h1)./1e9, Iac(fb2_h1).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 2: Fundamental")
force0y;
xlim(fb2);
setxtick(base_tick, false);

subplot(3, 1, 2);
hold off;
plot(freq_list(fb2_h2)./1e9.*2, Iac(fb2_h2).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 2: 2nd Harmonic")
force0y;
xlim(fb2.*2);
setxtick(base_tick*2, false);

subplot(3, 1, 3);
hold off;
plot(freq_list(fb2_h3)./1e9.*3, Iac(fb2_h3).*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', lw, 'Color', c1, 'MarkerSize', mksz);
hold on;
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Frequency Band 2: 3rd Harmonic")
legend("Meas.");
force0y;
xlim(fb2.*3);
setxtick(base_tick*3, false);

%% Adjust colors so my eyes stop burning :( 

c1 = [255, 242, 29]./255; % Bright yellow point in 'screen-used-lcars'
c2 = [137, 125, 233]./255; %  % Darker purle grid in 'screen-used-lcars'
c3 = [150, 222, 221]./255; % Blueish from top circle in 208897-20
c4 = [202, 129, 1]./255; % Brightestst orange in the circle aroudn the blue circle (c3), both in top, of 208897-20.

c_grid = [212, 217, 255]./255; % Lighter purpley gray from bigger circle around planet in 'screen-used-lcars'
window_frame_color = [.1, .1, .17]; % Dark color for window background
legend_color = [.6, .6, .6]; % Grey color for legend background

background_color = [0.4, 0.6, 0.4];
window_frame_color = [0.25, 0.25, 0.3];
c_grid = [137, 125, 233]./255;
axes_color = c3;
title_color = c3;

background_color = [0.3, 0.3, 0.35]; % COlor of plot area

for fig_no = [1, 2]
	
	figure(fig_no);
	
	% Configure window
	set(gcf, 'color', window_frame_color);
	
	for plot_no = 1:3
		
		subplot(3, 1, plot_no);
		
		%Turn background off
		set(gca, 'visible', 'on');
		set(gca, 'color', background_color);

		% Configure grid
		set(gca, 'GridColor', c_grid);
		set(gca, 'GridAlpha', .3); % Default is .15

		% Set axes label/ticks colors
		set(gca, 'YColor', axes_color);
		set(gca, 'XColor', axes_color);
		
		% CHagne title color
		title_old = get(gca, 'Title');
		title_old.Color = title_color;
		set(gca, 'Title', title_old);
		
		% Before inadvertently creating the legend, check if a legend is
		% present
		hide_legend = true;
		try
			if get(gca, 'Legend').Visible
				hide_legend = false;
			end
		catch
			%Do nothing
		end
		
		% Set legend color
		lgnd = legend(gca);
		set(lgnd, 'Color', legend_color);
		if hide_legend
			set(lgnd, 'Visible', 'off');
		end
	end
end

return

%% Zoom standard - scaled BW

figure(1);
for pn = 1:3
	subplot(3, 1, pn);
	xlim(fb1.*pn);
	setxtick(base_tick*pn, false);
end

figure(2);
for pn = 1:3
	subplot(3, 1, pn);
	xlim(fb2.*pn);
	setxtick(base_tick*pn, false);
end

%% Zoom center - const BW

figure(1);
for pn = 1:3
	subplot(3, 1, pn);
	xlim(fb1+mean(fb1).*(pn-1));
	setxtick(base_tick, false);
end

figure(2);
for pn = 1:3
	subplot(3, 1, pn);
	xlim(fb2+mean(fb2).*(pn-1));
	setxtick(base_tick, false);
end








