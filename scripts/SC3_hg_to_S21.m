% From 'hg_to_L0.m' but references power levels to signal generator as
% opposed to 'a1' of VNA. 
%
% Previously named: hg_to_L0_SGref_allpwrs.m
%
% Looks at harmonic generation (hg) of 2nd harmonic, then references it to
% the signal generator power (assumed -10 dBm) instead of the b1 signal
% from the VNA. This gives an actual S21 at the 2nd harmoinc. 

% Import data
load(dataset_path("DS5_FinePower_PO-1.mat"));
ds = ld;

pwr_all = unique([ld.SG_power_dBm]); % [dBm]
displ("Power Levels: ", pwr_all);

% Nathan's assumed parameters
len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

% Create conditoins
c = struct('SG_power', -10);
c.Vnorm = 2e-3;

pwr_VNA_dBm = -10; %[dBm]

figure(3);
hold off;

figure(4);
hold off;

figure(5);
hold off;

legend_list = {};

% Loop over each power level
CM = resamplecmap(colormap('parula'), numel(pwr_all));
idx = 0;
for pwr = pwr_all
	idx = idx + 1;
	
	% Filter Data - Generate Plot: Power vs DC bias

	% Set SG power parameter
	c.SG_power = pwr;

	% Calculate harmonics over bias sweep
	[harms, norm, Vdcs] = getHarmonicSweep(ld, c);
	h2 = harms.h2;
	Ibias = Vdcs.*iv_conv;

	% Convert VNA's funky units to real units
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	a_SG = sqrt(cvrt(pwr, 'dBm', 'W'));
	S21 = abs(h2).*a2./a_SG;
	S21_dB = lin2dB(S21);
	
	% Appoximate total current
	
	Ibequiv = Ibias + sign(Ibias).*sqrt(2*cvrt(pwr, 'dBm', 'W')/105)*0.707;
	
	V_port2 = S21.*a_SG.*sqrt(50);
	
	figure(3);
	plot(Ibias, S21_dB, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	figure(4);
	plot(Ibias, V_port2, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	figure(5);
	plot(Ibequiv, V_port2, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;

	legend_list = [legend_list(:)', {strcat("P = ", num2str(pwr), " dBm")}];
end



figure(3);
xlabel("Bias Current (A)");
ylabel("S_{21}");
title(strcat("2nd Harmonic Measurement, 10 GHz"));
grid on;
legend(legend_list{:},'NumColumns',2,'FontSize',8);
% set(hleg,'Location','best');
force0y;

figure(4);
xlabel("Bias Current (A)");
ylabel("V_{VNA} (V)");
title(strcat("2nd Harmonic Measurement, 10 GHz"));
grid on;
legend(legend_list{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;

figure(5);
xlabel("Zero-RF Power Equivalent Bias Current (A)");
ylabel("V_{VNA} (V)");
title(strcat("2nd Harmonic Measurement, 10 GHz"));
grid on;
legend(legend_list{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;













