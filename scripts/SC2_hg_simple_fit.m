% From 'hg_analyze.m'
% Previously named: hg_simple_check2.m
%
% Plots the 2nd harmonic data from a power sweep and performs a linear fit.

%% Import Data

% Import data
load(dataset_path("DS5_FinePower_PO-1.mat"));

ds = ld;

displ("Power Levels: ", unique([ld.SG_power_dBm]));
pwr = -6; % [dBm]

%% Filter Data - Generate Plot: Power vs DC bias

% Create conditoins
c = struct('SG_power', pwr, 'Vdc', 0.25);
c.harms = unique([ld.harmonic]);
c.Vnorm = 2e-3;

% Get DC Voltages from data
Vdcs = unique([ld.offset_V]);

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

R_rf = 100; % [Ohms] Conversion from RF power to current

q = 0.190;
omega = 2.*pi.*10e9;
P_rf = cvrt(-10, 'dBm', 'W');
Idc = Vdcs.*iv_conv;
Ipp = sqrt(P_rf./R_rf);
dIdt = Ipp.*omega.*sqrt(2)./2;

% Linear fit (only fits positive half of data, plots over entire range)
Idc_pos = Idc(ceil(numel(Idc)/2:end));
h2_pos = h2(ceil(numel(h2)/2:end));
P_pos = polyfit(Idc_pos,abs(h2_pos),1);
h2_fit = polyval(P_pos,Idc);

figure(5);
hold off;
scatter(Idc, abs(h2), mkz_a, 'Marker', mks, 'MarkerEdgeColor', c2, 'MarkerFaceColor', c2);
hold on;
plot(Idc, abs(h2_fit), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', c1);
xlabel("Bias Current (A)");
ylabel("Harmonic Power, b1/a2");
title(strcat("2nd Harmonic Measurement, ", num2str(pwr), " dBm, 10 GHz"));
grid on;
legend("Measured", "Fit");

displ("Plotted power level: ", pwr, " dBm");
displ();
displ("Linear fit (of positive side data):");
displ("  m  = ", P_pos(1));
displ("  Y0 = ", P_pos(2));












