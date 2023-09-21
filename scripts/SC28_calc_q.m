%% SC28
%
% Accepts a beta sweep and plots q estimates, tables, distribution etc.
%
% Previously named DS51
%
% Built off of SC1

DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','Main_Sweeps');
DATA_PATH2 = fullfile('/','Volumes','NO NAME', 'NIST September data');

% filename = 'beta_12Sept2023.mat'; % This is the file with the disconnected bias cable and thus flat response.
filename = 'beta_14Sept2023_1600.mat';

load(fullfile(DATA_PATH2, filename));

%%-- Convert V2 Data format to expected format for SC1

P_RF = 0;

% analysis_freqs = [1, 5, 10, 15].*1e9; % Frequencies to plot [GHz]

analysis_freqs = ds.configuration.frequency;

% analysis_freqs = (1:1:8).*1e9;
% invmask = (analysis_freqs == 4e9);
% analysis_freqs = analysis_freqs(~invmask);

% analysis_freqs = [1, 2, 3, 6].*1e9;

LIMIT_BIAS = true;

bias_max_A = 0.01;
bias_min_A = -0.01;

%%-----------------------------

% This script is built-off phase_to_x but uses correctData().
% Previously named: VNA_fundamental2_30March.m
%
% Analyzes phase data with I^2/q^2 model
%
% Note: this datasweep is looking at how the 

len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

CM = resamplecmap('parula', numel(analysis_freqs));

% Process each frequency
for f = analysis_freqs
	
	struct_key = string(strcat('f', num2str(f/1e9)));
	
	% Apply drift correction, unwrap phase, and normalize
	[cp, vb] = getCorrectedPhase(ds, P_RF, f, false);
	
	% Get bias currents
	Ibias = iv_conv.*vb;
	
	% Limit bias
	if LIMIT_BIAS
		
		Ipass = (Ibias >= bias_min_A) & (Ibias <= bias_max_A);
		cp = cp(Ipass);
		vb = vb(Ipass);
		Ibias = Ibias(Ipass);
		
	end
	
	% Remove zero point (don't divide by zero)
	I_calc = Ibias(cp ~= 0);
	cp_calc = cp(cp ~= 0);
	
	% Calculate q
	q = sqrt(180.*f.*len./abs(cp_calc)./Vp0).*abs(I_calc);
	
	% Save to structs
	q_vals.(struct_key) = q;
	I_vals.(struct_key) = Ibias;
	phase_vals.(struct_key) = cp;
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
idx = 0;
for f = analysis_freqs
	idx = idx + 1;

	% Get key
	struct_key = string(strcat('f', num2str(f/1e9)));
	
	% Get data
	q = q_vals.(struct_key);
	Ibias = I_vals.(struct_key);
	phase = phase_vals.(struct_key);
	
	% Add all qs
	all_qs = [all_qs, q];
	avg_q = [avg_q, mean(q)];
	std_q = [std_q, std(q)];
	
	% Add to graphs
	figure(1);
	plot(Ibias, phase, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz, 'Color', CM(idx, :));
	hold on;
	
	figure(2);
	plot(Ibias(phase ~= 0), q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz, 'Color', CM(idx, :));
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
plot(analysis_freqs./1e9, avg_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
hold on;
plot(analysis_freqs./1e9, std_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Frequency (GHz)");
ylabel("q (A)");
title("Nonlinearity over frequency");
legend("Mean", "Standard Deviation");

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