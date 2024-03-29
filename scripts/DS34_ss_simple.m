%% System Parameters

P_rf_gen_dBm = 0;
Ibias_A = linspace(-30, 30, 61).*1e-3;

P_rf_gen = cvrt(P_rf_gen_dBm, 'dBm', 'W');
Vgen = sqrt(P_rf_gen*200);

l_phys = 0.5;
C_ = 121e-12;
L0 = 900e-9;
q = 0.190;

f1 = 10e9;
ZL = 50;
Zsrc = 50;

Iac_conv = 1e-4;
conv_param1 = 0.5; % 0 = don't change guess, 1 = use new result as guess
conv_param2 = 0.75; % How much to reduce conv_param1 when conversion fails


%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

iv_conv = 9.5e-3; % A/V
c = struct('SG_power', -10);
c.Vnorm = 2e-3;
c.SG_power = P_rf_gen_dBm;

% Calculate harmonics over bias sweep
[harms, norm, Vdcs] = getHarmonicSweep(ld, c);
fund = harms.h1;
h2 = harms.h2;
Ibias = Vdcs.*iv_conv;

lw = 1.5;
mks = 10;

figure(1);
hold off;
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h1)), 'LineStyle', ':', 'LineWidth', lw, 'Marker', '+', 'Color', [0, 0, .7], 'MarkerSize', mks);
hold on;
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h2)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h3)), 'LineStyle', '-.', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.7, 0, 0], 'MarkerSize', mks);
grid on;
legend("Fundamental");
xlabel("DC Bias Current (mA)");
ylabel("Harmonic Power (dBm)");

%% For each datapoint, calclate bias condition

Imins = [];
Imaxs = [];
Iavgs = [];
iterations = [];

idx = 0;
for Vdc = Vdcs
	idx = idx + 1;
	
	Idc = Vdc.*iv_conv;
	
	% Set initial guess as DC alone
	Itot = Idc;
	iter = 0;
	
	% Iterate until current converges
	while true
		iter = iter + 1;
		
		% Calcualte inductance
		Ltot = L0*(1+Itot^2./q^2);
		
		% Calculate Vp, wavelength, etc
		Vp = 1./sqrt(Ltot.*C_);
		lambda = Vp/f1;
		theta_f1 = l_phys./lambda.*2.*pi;
		Z0 = sqrt(Ltot./C_);
		Zin = xfmr2zin(Z0, ZL, theta_f1);
		
		% Find transmitted power
		P0 = 1./2.*Vgen.^2.*real(Zin)./( (real(Zin) + real(Zsrc)).^2 + (imag(Zin) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
		
		% Find current range
		Zline1 = xfmr2zin(Z0, ZL, 0);
		Zline2 = xfmr2zin(Z0, ZL, 3.1415926535/2);
		[Iline1, ~] = ZPwr2IV(Zline1, P0);
		[Iline2, ~] = ZPwr2IV(Zline2, P0);
		
		% Find average current
		Iavg = (abs(Iline1) + abs(Iline2))./2;
		
		% Check for convergence
		if (abs(Iavg) - abs(Itot)-Idc) <= Iac_conv
			break;
		else
			delta = abs(Iavg) - abs(Itot);
			Itot = abs(Itot) + delta*conv_param1;
		end
		
	end
	% Add to data arrays
	Imins(idx) = min([Iline1, Iline2]);
	Imaxs(idx) = max([Iline1, Iline2]);
	Iavgs(idx) = Iavg;
	iterations(idx) = iter;
	
end

%% Plot results

figure(2);






%% Function Definitions

function pwr_w = VNA2W(harm)
	% Converts the VNA harmonic data to W
	
	a2 = sqrt(cvrt(-10, 'dBm', 'W'));
	
	pwr_w = (abs(harm).*a2).^2;
	
end

function pwr_dBm = VNA2dBm(harm)
	% Converts the VNA harmonic data to dBm
	
	pw = VNA2W(harm);
	pwr_dBm = cvrt(pw, 'W', 'dBm');
	
end

function [I, V] = ZPwr2IV(Zline, P_W)
	% Accepts an impedance (can be an impedance at a certain point along a
	% line) and returns the V and I coefficients (including phase) that
	% delivers P_W watts.
	
	% Find angle
	Z_arg = angle(Zline);
	
	% Return V and I
	I = sqrt(2.*P_W./cos(Z_arg)./Zline);
	V = sqrt(2.*P_W./cos(Z_arg).*Zline);
	
	
end