% This script is built-off phase_to_x but uses correctData().
% Previously named: VNA_fundamental2_30March.m
%
% Analyzes phase data with I^2/q^2 model
%
% Note: this datasweep is looking at how the 

len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

load(dataset_path("DS1_FOS-1.mat"));

% analysis_freqs = [1, 5, 50].*1e9; % Frequencies to plot [GHz]
% analysis_freqs = [1, 5, 10, 15].*1e9; % Frequencies to plot [GHz]
% analysis_freqs = [1,2, 3, 4, 5, 10, 15, 20, 30, 40, 50].*1e9; % Frequencies to plot [GHz]
analysis_freqs = (1:50).*1e9;

remove_outliers = true;
f_remove =[11, 13, 15, 16, 24, 28, 31, 32, 35, 36, 37, 41, 43, 44, 45, 46, 47, 48, 49].*1e9;
if remove_outliers
	for fr = f_remove
		idx_del = analysis_freqs == fr;
		if ~any(idx_del)
			warning("Missed point");
		end
		analysis_freqs = analysis_freqs(~idx_del);
	end
end

dp_idx = datapoint_index;
phase = Pb1a2;
vbias = Vbias;

% Get unique frequencies
frequencies = unique(freq);

% Process each frequency
for f = analysis_freqs
	
	% Find indecies for this frequency
	fIdx = (freq == f);

	struct_key = string(strcat('f', num2str(f/1e9)));
	
	dp_idx_s = dp_idx(fIdx);
	vbias_s = vbias(fIdx);
	phase_s = phase(fIdx);
	
	% Apply drift correction, unwrap phase, and normalize
	[cp, vb] = correctData(phase_s, vbias_s, dp_idx_s, f, true, true, false);
	
	% Get bias currents
	I = iv_conv.*vb;
	
	% Remove zero point (don't divide by zero)
	I_calc = I(cp ~= 0);
	cp_calc = cp(cp ~= 0);
	
	% Calculate q
	q = sqrt(180.*f.*len./abs(cp_calc)./Vp0).*abs(I_calc);
	
	% Save to structs
	q_vals.(struct_key) = q;
	I_vals.(struct_key) = I;
	phase_vals.(struct_key) = cp;
	
	
% 	figure(1);
% 	hold off;
% 	plot(I.*1e3, cp, 'Marker', '*', 'LineStyle', ':');
% 	grid on;
% 	xlabel("Bias Current (mA)");
% 	ylabel("\Delta Phase (deg)");
% 	title(strcat("Phase Change from Zero-Bias, f=", num2str(f/1e9), " GHz"));
% 	
% 	figure(2);
% 	hold off;
% 	plot(I_calc.*1e3, q, 'Marker', '*', 'LineStyle', ':');
% 	grid on;
% 	xlabel("Bias Current (mA)");
% 	ylabel("q (A)");
% 	force0y;
% 	title(strcat("Nonlinearity Estimate, f=", num2str(f/1e9), " GHz"));
	
% 	input("Enter to continue");

end

%% Show summary

lw = 1.5;
mkz = 10;

figure(1);
hold off;

figure(2);
hold off;

mt = MTable();
mt.title("Nonlinearity Summary");
mt.row(["Freq (GHz)", "avg(q) [mA]", "min(q) [mA]", "max(q) [mA]", "stdev(q) [mA]"]);

legend_vals = {};

all_qs = [];
avg_q = [];
std_q  = [];

% Process each frequency
for f = analysis_freqs

	% Get key
	struct_key = string(strcat('f', num2str(f/1e9)));
	
	% Get data
	q = q_vals.(struct_key);
	I = I_vals.(struct_key);
	phase = phase_vals.(struct_key);
	
	% Add all qs
	all_qs = [all_qs, q];
	avg_q = [avg_q, mean(q)];
	std_q = [std_q, std(q)];
	
	% Add to graphs
	figure(1);
	plot(I, phase, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
	hold on;
	
	figure(2);
	plot(I(phase ~= 0), q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
	hold on;
	
	% Add to table
	mt.row([string(num2str(f/1e9)), string(num2str(mean(q.*1e3))), string(num2str(min(q.*1e3))), string(num2str(max(q.*1e3))), string(num2str(std(q.*1e3))) ]);
	
	% Add to legend
	legend_vals = [legend_vals(:)', {strcat(num2str(f/1e9), " GHz")}];
end

all_q_mean = mean(all_qs)*1e3;
all_q_std = std(all_qs)*1e3;

% Print table
disp(mt.str());
displ(newline, "Total Average: ", all_q_mean, " mA");
displ("Total St.Dev.: ", all_q_std, " mA");

% Finish graphs
figure(1);
grid on;
legend(legend_vals{:});
title("Phase Change from Zero-Bias");
xlabel("Bias Current (mA)");
ylabel("\Delta Phase (deg)");

c_std = [1, 1, 1].*0.3;
c_mean = [0, 0, 0];
figure(2);
grid on;
legend(legend_vals{:});
title("Nonlinearity Estimate");
xlabel("Bias Current (mA)");
ylabel("q (A)");
hlin(all_q_mean/1e3, 'LineStyle', '--', 'Color', c_mean);
hlin((all_q_mean+all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);
hlin((all_q_mean-all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);

% Plot over frequency
figure(3);
hold off;
errorbar(analysis_freqs./1e9, avg_q.*1e3, std_q.*1e3, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
hold on;
% plot(analysis_freqs./1e9, std_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
hlin(mean(avg_q), 'Color', [0.2, 0.2, 0.2], 'LineStyle', '--');
grid on;
xlabel("Frequency (GHz)");
ylabel("q (mA)");
title("Nonlinearity over frequency");
% legend("Mean", "Standard Deviation");
force0y;

% Plot histogram
sel_qs = all_qs(all_qs < 0.4);
figure(4);
hold off;
histogram(sel_qs, 'FaceColor', [0, 0, .7]);
hold on;
vlin(mean(sel_qs), "LineStyle", '--', 'Color', c_mean);
vlin(mean(sel_qs)-std(sel_qs), "LineStyle", ':', 'Color', c_std);
vlin(mean(sel_qs)+std(sel_qs), "LineStyle", ':', 'Color', c_std);
xlabel("q Value (A)");
ylabel("Counts");
title("Distribution of q Estimates");