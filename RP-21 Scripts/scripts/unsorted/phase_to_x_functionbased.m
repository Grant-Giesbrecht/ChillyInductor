% This script is built-off phase_to_x but uses correctData().
%
%

load(fullfile("","","FOS1_Data.mat"));

plot_freqs = [1, 5, 50].*1e9; % Frequencies to plot [GHz]

dp_idx = datapoint_index;
phase = Pb1a2;
vbias = Vbias;

% Get unique frequencies
frequencies = unique(freq);

% Process each frequency
for f = plot_freqs
	
	% Find indecies for this frequency
	fIdx = (freq == f);
	
	struct_key = string(strcat('f', num2str(f/1e9)));
	
	dp_idx_s = dp_idx(fIdx);
	vbias_s = vbias(fIdx);
	phase_s = phase(fIdx);
	
	[cp, vb] = correctData(phase_s, vbias_s, dp_idx_s, f, true, true, true);
	
	figure(17);
	hold off;
	plot(vb, cp, 'Marker', '*', 'LineStyle', ':');
	grid on;
	
	input("Enter to continue");

end
