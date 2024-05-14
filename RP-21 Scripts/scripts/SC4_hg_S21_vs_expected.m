% From 'hg_to_L0.m' but references power levels to signal generator as
% opposed to 'a1' of VNA. 
%
% Previously named: hg_to_L0_SgR_compare.m
%
% Extends the functionality of SC3 but gets rid of the 'equivalent bias'
% graph (which didn't explain the seen phenomenon) and instead tries to
% calculate the expected 2nd harmonic strength and compares measurement to
% expectation to try to gauge system loss.

% Import data
load(dataset_path("DS5_FinePower_PO-1.mat"));

ds = ld;

pwr_all = unique([ld.SG_power_dBm]); % [dBm]
displ("Power Levels: ", pwr_all);

% Nathan's assumed parameters
len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

f = 10e9;

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

figure(6);
hold off;

figure(7);
hold off;

legend_list = {};
legend_list2 = {};

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
	fund = harms.h1;
	h2 = harms.h2;
	Ibias = Vdcs.*iv_conv;
	
	% Convert VNA's funky units to real units (2nd harmonic)
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	a_SG = sqrt(cvrt(pwr, 'dBm', 'W'));
	S21 = abs(h2).*a2./a_SG;
	S21_dB = lin2dB(S21);
	
	% Approximate Ipp from fundamental
	S21_fund = abs(fund).*a2./a_SG;
	V_port_fund = S21_fund.*a_SG.*sqrt(50);
	Ipp_fund = V_port_fund./50;
	
	% Calculate things to plot
	V_port2 = S21.*a_SG.*sqrt(50);
	P_rec = (S21.*a_SG).^2;
	
	% Calculate Ipp from power (this was original method and wrong)
	w = 2.*3.14159.*f;
	j = complex(0, 1);
	L0 = 1e-6*0.5;
	L0_guess = 4.5e-9;
% 	L0_guess = 0.9e-9;
	ZL_est = L0_guess*j*w;
	Idc = abs(Ibias);
	Ipp_theory = abs( 2/sqrt(2)*sqrt(cvrt(pwr, 'dBm', 'W')/(105+ZL_est)) );
	
	% Approximate Ipp from generator voltage
	Vgen = sqrt(cvrt(pwr, 'dBm', 'W')*200);
	Ipp_vg = abs(Vgen/(105+ZL_est));

	% Calculate expected voltage
	Ipp = mean(Ipp_fund);
% 	Ipp = Ipp_theory;
% 	Ipp = Ipp_vg;
	q = .19;
	f2w = L0./q.^2.*(Idc*Ipp^2*w);
	f2w_vna = f2w/2; % Divide by 2, voltage across L split between 2 loads (VNA and SG)
	P_est = f2w.*Ipp;
	
	
	% Calculate attenuation in dB
	atten = lin2dB(V_port2./f2w_vna);
% 	atten = lin2dB(P_rec./P_est);
	
	figure(3);
	plot(Ibias, S21_dB, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	figure(4);
	plot(Ibias, V_port2, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;

	legend_list = [legend_list(:)', {strcat("P = ", num2str(pwr), " dBm")}];
	
	%skip the baddies
	if pwr > 7
		continue
	end
	
	displ("Ipp Estimates: ( P = ", pwr, " dBm)")
	displ("  From PWR & Z: |Ipp| = ", Ipp_theory*1e3, " mA");
	displ("  From Vgen   : |Ipp| = ", Ipp_vg*1e3, " mA");
	displ("  From Fund.  : min|Ipp| = ", min(abs(Ipp_fund))*1e3, " mA");
	displ("     ...      : max|Ipp| = ", max(abs(Ipp_fund))*1e3, " mA");
	displ("     ...      : avg|Ipp| = ", mean(abs(Ipp_fund))*1e3, " mA");
	
	figure(5);
	plot(Ibias, f2w_vna, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;

	figure(6);
	plot(Ibias, atten, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
	
	legend_list2 = [legend_list2(:)', {strcat("P = ", num2str(pwr), " dBm")}];
	
	figure(7);
	plot(Ibias, Ipp_fund, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
	hold on;
% 	
% 	figure(8);
% 	plot(Ibias, atten, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
% 	hold on;
	
	
	
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
xlabel("Bias Current (A)");
ylabel("Voltage (V)");
title(strcat("Expected VNA Voltage - 2nd Harmonic, 10 GHz"));
grid on;
legend(legend_list2{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;

figure(6);
xlabel("Bias Current (A)");
ylabel("Attenuation (dB)");
title(strcat("2nd Harmonic Measurment relative to Expected, 10 GHz"));
grid on;
legend(legend_list2{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;

figure(7);
xlabel("Bias Current (A)");
ylabel(" I_{PP} Fundamental(dB)");
title(strcat("Fundamental Appox. Current, 10 GHz"));
grid on;
legend(legend_list2{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;






