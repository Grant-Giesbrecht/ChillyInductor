% This script is built-off phase_to_x but uses correctData().
%
%

reselect_frequencies = false;

len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

load(fullfile("","","FOS1_Data.mat"));

% analysis_freqs = [1, 5, 50].*1e9; % Frequencies to plot [GHz]
% analysis_freqs = [1, 5, 10, 15].*1e9; % Frequencies to plot [GHz]
% analysis_freqs = [1,2, 3, 4, 5, 10, 15, 20, 30, 40, 50].*1e9; % Frequencies to plot [GHz]
analysis_freqs = (1:50).*1e9;

% Load frequencies
load("keep_freqs_full.mat");
analysis_freqs = keep_freqs;
% analysis_freqs = border_freqs;
% analysis_freqs = trash_freqs;

dp_idx = datapoint_index;
phase = Pb1a2;
vbias = Vbias;

% Get unique frequencies
frequencies = unique(freq);

% Define output arrays
o4fit_q = zeros(1, numel(analysis_freqs));
o4fit_x = zeros(1, numel(analysis_freqs));
o4fit_R2 = zeros(1, numel(analysis_freqs));
o4fit_rmse = zeros(1, numel(analysis_freqs));

% Process each frequency
idx = 0;

keep_freqs = [];
trash_freqs = [];
border_freqs = [];

for f = analysis_freqs
	idx = idx + 1;
	
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
	
	% Estimate q
	q_guess = sqrt(180.*f.*len./abs(cp_calc)./Vp0).*abs(I_calc);
	mqg = mean(q_guess);
	
	% Create fitting function object
	func = @(q, x, Idata) 360.*len./Vp0.*f.*(Idata.^2./2./q.^2 - Idata.^4./8./q.^4 + Idata.^4./2./x.^4);
% 	func = @(q, x, Idata) 360.*5.8e-9.*1e9.*(Idata.^2./2./q.^2 - Idata.^4./8./q.^4 + Idata.^4./2./x.^4);
	ft_func = fittype(func, 'coeff', {'q', 'x'}, 'independent', {'Idata'});
	
	% Create fit options
	x0 = [mqg, 1e-3];
	fopt = fitoptions('Method', 'NonlinearLeastSquares', 'Upper', [1, .2], 'Lower', [1e-6, 1e-6], 'StartPoint', x0);
	
	% Perform fit
	
	[curve_fit, gof] = fit(I_calc', abs(cp_calc)', ft_func, fopt);
	coefs = coeffvalues(curve_fit);
	
	data_color = [0, .8, 0];
	data_color2 = [0, .2, 0];
	order2_color = [.7, 0, 0];
	order4_color = [0, 0, .7];
	lw = 1.5;
	mkzs = 70;
	mkz_list = ones(1, numel(cp_calc)).*mkzs;
	
	figure(5);
	hold off;
	scatter(I_calc.*1e3, abs(cp_calc), mkz_list, 'Marker', 'o', 'MarkerFaceColor', data_color, 'MarkerEdgeColor', data_color2);
	hold on;
	plot(I_calc.*1e3, 180.*f.*len./Vp0.*I_calc.^2./mqg.^2, 'LineStyle', '--', 'Color', order2_color, 'LineWidth', lw);
	plot(I_calc.*1e3, curve_fit(I_calc), 'LineStyle', '-.', 'Color', order4_color, 'LineWidth', lw);
	legend("Data", "2nd Order Fit", "4th Order Fit", 'Location', 'SouthEast');
	grid on;
	xlabel("Bias Current (mA)");
	ylabel("\Delta Phase (deg)");
	title(strcat("Fitting Function Comparison, f=", num2str(f/1e9), " GHz"));
	
	barprint(strcat("f = ", num2str(f/1e9), " GHz"));
	displ("q = ", coefs(1)*1e3, " mA");
	displ("x = ", coefs(2)*1e3, " mA");
	displ("R^2 = ", gof.rsquare, " ");
	displ("rmse = ", gof.rmse, " ");
	
	if reselect_frequencies
		displ("Save, delete, borderline?");
		while true
			commandwindow;
			keep = input("::", 's');
			if keep == 's'
				displ("--> Save");
				keep_freqs = [keep_freqs, f];
				break;
			elseif keep == 'd'
				displ("Skip (trash)");
				trash_freqs = [trash_freqs, f];
				break;
			elseif keep == 'b'
				displ("Skip (borderline)");
				border_freqs = [border_freqs, f];
				break;
			elseif keep == 'exit'
				displ("Keep: [",keep_freqs,"]");
				displ("Border: [", border_freqs,"]");
				displ("Keep: [", trash_freqs,"]");
				save('keep_freqs.mat', 'keep_freqs', 'trash_freqs', 'border_freqs');
				return
			end
		end		
	end	
	
	% Save to structs
	q_vals.(struct_key) = q_guess;
	I_vals.(struct_key) = I;
	phase_vals.(struct_key) = cp;
	
	% Save fit data
	o4fit_q(idx) = coefs(1);
	o4fit_x(idx) = coefs(2);
	o4fit_r2(idx) = gof.rsquare;
	o4fit_rmse(idx) = gof.rmse;

end

if reselect_frequencies
	displ("Keep: [",keep_freqs,"]");
	displ("Border: [", border_freqs,"]");
	displ("Keep: [", trash_freqs,"]");
	save('keep_freqs.mat', 'keep_freqs', 'trash_freqs', 'border_freqs');
end

% %% Show summary
% 
% lw = 1.5;
% mkz = 10;
% 
% figure(1);
% hold off;
% 
% figure(2);
% hold off;
% 
% mt = MTable();
% mt.title("Nonlinearity Summary");
% mt.row(["Freq (GHz)", "avg(q) [mA]", "min(q) [mA]", "max(q) [mA]", "stdev(q) [mA]"]);
% 
% legend_vals = {};
% 
% all_qs = [];
% avg_q = [];
% std_q  = [];
% 
% % Process each frequency
% for f = analysis_freqs
% 
% 	% Get key
% 	struct_key = string(strcat('f', num2str(f/1e9)));
% 	
% 	% Get data
% 	q = q_vals.(struct_key);
% 	I = I_vals.(struct_key);
% 	phase = phase_vals.(struct_key);
% 	
% 	% Add all qs
% 	all_qs = [all_qs, q];
% 	avg_q = [avg_q, mean(q)];
% 	std_q = [std_q, std(q)];
% 	
% 	% Add to graphs
% 	figure(1);
% 	plot(I, phase, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
% 	hold on;
% 	
% 	figure(2);
% 	plot(I(phase ~= 0), q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
% 	hold on;
% 	
% 	% Add to table
% 	mt.row([string(num2str(f/1e9)), string(num2str(mean(q.*1e3))), string(num2str(min(q.*1e3))), string(num2str(max(q.*1e3))), string(num2str(std(q.*1e3))) ]);
% 	
% 	% Add to legend
% 	legend_vals = [legend_vals(:)', {strcat(num2str(f/1e9), " GHz")}];
% end
% 
% all_q_mean = mean(all_qs)*1e3;
% all_q_std = std(all_qs)*1e3;
% 
% % Print table
% disp(mt.str());
% displ(newline, "Total Average: ", all_q_mean, " mA");
% displ("Total St.Dev.: ", all_q_std, " mA");
% 
% % Finish graphs
% figure(1);
% grid on;
% legend(legend_vals{:});
% title("Phase Change from Zero-Bias");
% xlabel("Bias Current (mA)");
% ylabel("\Delta Phase (deg)");
% 
% c_std = [1, 1, 1].*0.3;
% c_mean = [0, 0, 0];
% figure(2);
% grid on;
% legend(legend_vals{:});
% title("Nonlinearity Estimate");
% xlabel("Bias Current (mA)");
% ylabel("q (A)");
% hlin(all_q_mean/1e3, 'LineStyle', '--', 'Color', c_mean);
% hlin((all_q_mean+all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);
% hlin((all_q_mean-all_q_std)/1e3, 'LineStyle', '--', 'Color', c_std);
 
% Plot over frequency

mkz = 10;

figure(6);
subplot(2, 1, 1);
hold off;
plot(analysis_freqs./1e9, o4fit_q, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
hold on;
plot(analysis_freqs./1e9, o4fit_x, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Frequency (GHz)");
ylabel("q (A)");
legend("q", "x");

subplot(2, 1, 2);
hold off;
plot(analysis_freqs./1e9, o4fit_rmse, 'Marker', '+', 'LineStyle', ':', 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Frequency (GHz)");
ylabel("RMSE");

title("Nonlinearity Coefficients over frequency");

 
% % Plot histogram
% sel_qs = all_qs(all_qs < 0.4);
% figure(4);
% hold off;
% histogram(sel_qs, 'FaceColor', [0, 0, .7]);
% hold on;
% vlin(mean(sel_qs), "LineStyle", '--', 'Color', c_mean);
% vlin(mean(sel_qs)-std(sel_qs), "LineStyle", ':', 'Color', c_std);
% vlin(mean(sel_qs)+std(sel_qs), "LineStyle", ':', 'Color', c_std);
