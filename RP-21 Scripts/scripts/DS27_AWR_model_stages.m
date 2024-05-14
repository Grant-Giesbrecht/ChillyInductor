
files = ["C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\900_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\920_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\950_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\970_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1000_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1100_stages_processed.mat"];

num_stages = [900, 920, 950, 970, 1e3, 1.1e3];

% files = ["C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\900_stages_processed.mat", ...
% 	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1000_stages_processed.mat", ...
% 	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1100_stages_processed.mat"];
% 
% num_stages = [900, 1e3, 1.1e3];

files = ["C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\900_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\900_stages_50x_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1000_stages_processed.mat", ...
	"C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\1100_stages_processed.mat"];

num_stages = [900, 900, 1e3, 1.1e3];

% Prepare figures
figure(1);
hold off;

CM = resamplecmap(colormap('hot'), numel(files));
LL = {};
mkz = 10;
lnz = 1.5;
for i = 1:numel(files)
	
	% Load data
	load(files(i));
	
	mk = '+';
	if files(i) == "C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\Datasets\AWR_data\Inductor_Model2\Processed\900_stages_50x_processed.mat"
		mk = 'o';
	end
	
	% Plot it
	plot(I_DC_mA, cvrt(P2H_10GHz_dBW, 'dBW', 'dBm'), 'LineStyle', ':', 'Marker', mk, 'Color', CM(i, :), 'LineWidth', lnz, 'MarkerSize', mkz);
	hold on;
	
	% Add to legend
	LL = [LL(:)', {strcat(num2str(num_stages(i)), " stages")}];
end

xlabel("Bias Current (mA)");
ylabel("2nd Harmonic Power (dBm)");
title("Sensitivity to Number of AWR Simulation Stages - 10 GHz, P_{SG} = 4 dBm");
grid on;
legend(LL{:});
ylim([-75, 0]);

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

for fig_no = [1]
	
	figure(fig_no);
	
	% Configure window
	set(gcf, 'color', window_frame_color);
	
% 	for plot_no = 1:3
% 		
% 		subplot(3, 1, plot_no);
		
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
% 	end
end

return
