% Built off of SC2
%
% However instead of doing a linear fit, it calculated 'm' for each point,
% much like how 'q' was found in SC1.

%% Import Data

% Import data
load(dataset_path("DS5_FinePower_PO-1.mat"));

ds = ld;

all_pwrs = unique([ld.SG_power_dBm]);

all_pwrs = all_pwrs(all_pwrs < 1.9);

displ("Power Levels: ", all_pwrs);

%% Filter Data - Generate Plot: Power vs DC bias

% Get DC Voltages from data
Vdcs = unique([ld.offset_V]);

figure(1);
hold off;

figure(2);
hold off;

figure(3);
hold off;

figure(4);
hold off;

figure(6);
hold off;

figure(7);
hold off;

LL1 = {};
LL2 = {};
LL7 = {};

q = 0.190;
omega = 2.*pi.*10e9;

m_means = zeros(1, numel(all_pwrs));
m_means_sel = zeros(1, numel(all_pwrs));
m_stdevs = zeros(1, numel(all_pwrs));
I_mean = zeros(1, numel(all_pwrs));

RMSE_standard = zeros(1, numel(all_pwrs));
RMSE_equiv = zeros(1, numel(all_pwrs));

idx_meister = 0;
CM = resamplecmap(colormap('parula'), numel(all_pwrs));
for pwr = all_pwrs
	idx_meister = idx_meister + 1;
	
	% Create conditoins
	c = struct('SG_power', pwr, 'Vdc', 0.25);
	c.harms = unique([ld.harmonic]);
	c.Vnorm = 2e-3;

	% Plot at constant SG power, sweep offset
	num_normal = 0;
	h1 = zeros(1, numel(Vdcs));
	h2 = zeros(1, numel(Vdcs));
	h3 = zeros(1, numel(Vdcs));
	h4 = zeros(1, numel(Vdcs));
	h5 = zeros(1, numel(Vdcs));

	idx = 0;
	for vdc = Vdcs
		idx = idx + 1;

		% Get harmonics at this datapoint
		c.Vdc = vdc;
		[norm, harms, err] = getHarmonics(ds, c);

		% Skip points that went normal
		if norm.pf
			num_normal = num_normal + 1;
			harms = [NaN, NaN, NaN, NaN, NaN];
		end

		% Add to lists
		h1(idx) = harms(1);
		h2(idx) = harms(2);
		h3(idx) = harms(3);
		h4(idx) = harms(4);
		h5(idx) = harms(5);
		
	end	
	
	mkz_a = ones(1, numel(Vdcs)).*20;
	ls = ':';
	lw = 1.3;
	mks = 'o';

	ls2 = '--';
	lw2 = 1.3;
	mks2 = 'None';
	mkz2 = 8;

	c1 = [0, 0, .7];
	c2 = [.7, 0, 0];
	c3 = [0, .7, 0];
	c4 = [.5, .5, .5];

	% Nathan's assumed parameters
	len = 0.5; % meters
	Vp0 = 86.207e6; % m/s
	iv_conv = 9.5e-3; % A/V
	
	V_h2 = abs(h2).*sqrt(cvrt(-10, 'dBm', 'W')).*sqrt(50);
	
% 	R_rf = 100; % [Ohms] Conversion from RF power to current
	
	Idc = Vdcs.*iv_conv;
	
	
	% Get two different Iac estimates - one from Prf, one from P fund
	Ifund_meas_mA = abs(h1).*sqrt(cvrt(-10, 'dBm', 'W'))./sqrt(50).*1e3;
	Ifund_ideal_mA = zeros(1, numel(Idc))+sqrt(cvrt(pwr, 'dBm', 'W'))./sqrt(50).*1e3;
	
	Idc_equiv = Idc + sign(Idc+1e-9).*Ifund_meas_mA./1e3;
	
% 	P_rf = cvrt(-10, 'dBm', 'W');
	
% 	Idc_equiv = Idc + sqrt(cvrt(pwr, 'dBm', 'W'))./sqrt(50);
	
% 	Ipp = sqrt(P_rf./R_rf);
% 	dIdt = Ipp.*omega.*sqrt(2)./2;
	
	Ivalid = ~isnan(h2);
	Idc = Idc(Ivalid);
	h2 = h2(Ivalid);
	Idc_nz = Idc(Idc ~= 0);
	m = abs(V_h2(Idc ~= 0))./abs(Idc_nz);
	m_equiv = abs(V_h2)./abs(Idc_equiv);
	mkz_a = mkz_a(Ivalid);
	mkz_a_nz = mkz_a(Idc ~= 0);
	
	% Get stats about slope
	I_sel = (abs(Idc) < 0.02) & (abs(Idc) > 0.01);
	m_means_sel(idx_meister) = mean(m(I_sel));
	m_means(idx_meister) = mean(m);
	m_stdevs(idx_meister) = std(m);
	
	% Calculate RMSE
	RMSE_standard(idx_meister) = sqrt(mean(( abs(V_h2(Idc ~= 0)) - abs(Idc_nz.*mean(m)) ).^2));
	RMSE_equiv(idx_meister) = sqrt(mean(( abs(V_h2) - abs(Idc_equiv.*mean(m_equiv)) ).^2));
	
	figure(1);
	scatter(Idc.*1e3, abs(V_h2), mkz_a, 'Marker', mks, 'MarkerEdgeColor', CM(idx_meister, :), 'MarkerFaceColor', CM(idx_meister, :));
	hold on;
	plot(Idc_nz.*1e3, abs(Idc_nz.*mean(m)), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', CM(idx_meister, :));
	LL1 = [LL1(:)', {strcat("P = ", num2str(pwr), " dBm, Meas")}, {strcat("P = ", num2str(pwr), " dBm, Fit")}];
	displ("Pwr = ", pwr, " dBm, m = ", mean(m));
	
	figure(2);
	scatter(Idc_nz.*1e3, m, mkz_a_nz, 'Marker', mks, 'MarkerEdgeColor', CM(idx_meister, :), 'MarkerFaceColor', CM(idx_meister, :));
	LL2 = [LL2(:)', {strcat("P = ", num2str(pwr), " dBm")}];
	hold on;
	
	figure(3);
	scatter(Idc.*1e3, abs(V_h2), mkz_a, 'Marker', mks, 'MarkerEdgeColor', CM(idx_meister, :), 'MarkerFaceColor', CM(idx_meister, :));
	hold on;
	plot(Idc.*1e3, abs(Idc_equiv.*mean(m_equiv)), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', CM(idx_meister, :));
	displ("Pwr = ", pwr, " dBm, m = ", mean(m));
	
	figure(6);
	scatter(Idc_equiv.*1e3, abs(V_h2), mkz_a, 'Marker', mks, 'MarkerEdgeColor', CM(idx_meister, :), 'MarkerFaceColor', CM(idx_meister, :));
	hold on;
	plot(Idc_equiv.*1e3, abs(Idc_equiv.*mean(m_equiv)), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', CM(idx_meister, :));
	displ("Pwr = ", pwr, " dBm, m = ", mean(m));
	
	figure(4);
	scatter(Idc.*1e3, m_equiv, mkz_a, 'Marker', mks, 'MarkerEdgeColor', CM(idx_meister, :), 'MarkerFaceColor', CM(idx_meister, :));
	hold on;
	
	figure(7);
	plot(Idc, Ifund_meas_mA, 'LineStyle', '--', 'Marker', '+', 'Color', CM(idx_meister, :), 'LineWidth', 1.5);
	hold on;
	plot(Idc, Ifund_ideal_mA, 'LineStyle', ':', 'Marker', 'O', 'Color', CM(idx_meister, :), 'LineWidth', 1.0);
	LL7 = [LL7(:)', {strcat("P = ", num2str(pwr), " dBm, h2 est")}, {strcat("P = ", num2str(pwr), " dBm, ideal est")}];
	
	I_mean(idx_meister) = mean(h1.*cvrt(-10, 'dBm', 'W')./sqrt(50));
	I_mean(idx_meister) = mean(sqrt(cvrt(pwr, 'dBm', 'W'))./sqrt(50));
	
end

figure(1);
xlabel("Bias Current (mA)");
ylabel("Harmonic Voltage (V)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL1{:});

figure(2);
xlabel("Bias Current (mA)");
ylabel("Slope (\Omega)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL2{:});

figure(3);
xlabel("Bias Current (mA)");
ylabel("Harmonic Voltage (V)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL1{:});

figure(4);
xlabel("Bias Current (mA)");
ylabel("Slope (\Omega)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL2{:});

figure(5);
hold off;
plot(all_pwrs, m_means, 'LineStyle', ':', 'Marker', '+', 'Color', [0, 0, .6]);
xlabel("RF Power (dBm)");
ylabel("Slope (\Omega)");
title("Slope versus RF Power");
grid on;

figure(6);
xlabel("Bias + RF Current (mA)");
ylabel("Harmonic Voltage (V)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL1{:});

figure(7);
xlabel("Bias Current (A)");
ylabel("RF Current (mA)");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend(LL7{:});

figure(8);
hold off;
plot(all_pwrs, m_means./I_mean.^2.*q.^2./omega.*1e9, 'LineStyle', ':', 'Marker', '+', 'Color', [0, 0, .6], 'LineWidth', 1.5);
hold on;
plot(all_pwrs, m_means_sel./I_mean.^2.*q.^2./omega.*1e9, 'LineStyle', ':', 'Marker', '+', 'Color', [0, 0.6, 0], 'LineWidth', 1.5);
xlabel("RF Power (dBm)");
ylabel("L_0 (nH)");
title("Inductance: Calculated from m, \omega, and I_{AC}");
legend("Average: All bias vals", "Average: 10-20 mA");
grid on;
force0y;

figure(9);
hold off;
plot(all_pwrs, RMSE_standard.*1e3, 'LineStyle', ':', 'Marker', '+', 'Color', [0.8, 0, 0], 'LineWidth', 1.5);
hold on;
plot(all_pwrs, RMSE_equiv.*1e3, 'LineStyle', ':', 'Marker', '+', 'Color', [0.4, 0, 0.8], 'LineWidth', 1.5);
legend("I_{AC} = 0", "I_{AC} from fund.", 'Location', 'NorthWest');
title("Error Comparison of Different Slope Estimation Methods");
xlabel("RF Power (dBm)");
ylabel("RMSE (mV)");
grid on;
force0y;




