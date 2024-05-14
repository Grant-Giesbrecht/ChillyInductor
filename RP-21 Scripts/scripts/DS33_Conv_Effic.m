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

% Import loss data
load(fullfile('script_assets', 'SC4', 'cryostat_sparams.mat'))
S21_linear = dB2lin(S21_dB, 20);

ds = ld;

pwr_all = unique([ld.SG_power_dBm]); % [dBm]
displ("Power Levels: ", pwr_all);

% Nathan's assumed parameters
len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

f = 10e9;

S21_idx_h1 = findClosest(freq_Hz, f);
S21_idx_h2 = findClosest(freq_Hz, f*2);
S21_comp = (S21_linear(S21_idx_h1) + S21_linear(S21_idx_h2))/2;

% Create conditoins
c = struct('SG_power', -10);
c.Vnorm = 2e-3;

pwr_VNA_dBm = -10; %[dBm]

figure(1);
hold off;

figure(2);
hold off;

legend_list = {};
legend_list2 = {};

% Loop over each power level
CM = [0, 0, 0.6];
idx = 0;
pwr = 0;
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

% Calculate power at frequencies and ports
pwr_vna_h2 = (abs(h2).*a2).^2;
pwr_vna_h1 = (abs(fund).*a2).^2;
pwr_vna_h3 = (abs(harms.h3).*a2).^2;
pwr_vna_h4 = (abs(harms.h4).*a2).^2;
pwr_vna_h5 = (abs(harms.h5).*a2).^2;
pwr_sg_h1  = fund.*0 + cvrt(pwr, 'dBm', 'W');
pwr_sum = (pwr_vna_h1+pwr_vna_h2+pwr_vna_h3+pwr_vna_h4+pwr_vna_h5);

% Calculate conv. efficiency
conv_effic = pwr_vna_h2./pwr_sg_h1./S21_comp.*100;
conv_effic_est2 = pwr_vna_h2./pwr_sum.*100;

figure(1);
plot(Ibias, pwr_vna_h2.*1e3, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', [0.7, 0, 0.4]);
hold on;
plot(Ibias, pwr_sum.*1e3, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', [0.4, 0, 0.7]);

figure(2);
plot(Ibias, conv_effic, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
hold on;

figure(3);
plot(Ibias, conv_effic_est2, 'Marker', '+', 'LineStyle', ':', 'LineWidth', 1.3, 'Color', CM(idx,:));
hold on;


legend_list = [legend_list(:)', {strcat("P = ", num2str(pwr), " dBm")}];




figure(1);
xlabel("Bias Current (A)");
ylabel("Power (mW)");
title(strcat("VNA 2nd Harmonic Power"));
grid on;
legend('2nd Harmonic Power', 'Total Power');
% set(hleg,'Location','best');
force0y;

figure(2);
xlabel("Bias Current (A)");
ylabel("Conversion Efficiency (%)");
title(strcat("Conversion Efficiency; P_{VNA}(f=2*f_0)/P_{SG} "));
grid on;
legend(legend_list{:},'NumColumns',1,'FontSize',8);
% set(hleg,'Location','best');
force0y;

figure(3);
xlabel("Bias Current (A)");
ylabel("Conversion Efficiency (%)");
title(strcat("Conversion Efficiency; P_{VNA}(f=2*f_0)/P_{VNA}(f=:) "));
grid on;
legend(legend_list{:},'NumColumns',2,'FontSize',8);
% set(hleg,'Location','best');
force0y;





