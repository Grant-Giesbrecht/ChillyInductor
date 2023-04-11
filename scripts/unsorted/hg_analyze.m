% From 'analyze_harmonics.m'

%% Import Data

% Data File
datafolder = '';
filename = "PO1.mat";

% Import data
load(fullfile(datafolder, filename));

ds = ld;

%% Filter Data - Generate Plot: Power vs DC bias

% Create conditoins
c = struct('SG_power', -10, 'Vdc', 0.25);
c.harms = unique([ld.harmonic]);
c.Vnorm = 2e-3;

% Test function
[norm, harms, err] = getHarmonics(ds, c);


% Plot at constant SG power, sweep offset
Vdcs = linspace(-3.5, 3.5, 29);
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



%% Calculate Harmoincs



len = 0.5; % meters
Vp0 = 86.207e6; % m/s
iv_conv = 9.5e-3; % A/V

Idc = Vdcs.*iv_conv;
Ipp = 2/sqrt(2);

params = struct('Idc', Idc, 'Ipp', Ipp, 'q', 0.190, 'r', 0.113, 'L0',  7.0931e-15, 'f', 10e9, 'R_power', 1/iv_conv );

[f1w, f2w, f3w, f4w] = calc_harmonics(params);

%% Plot

ls = ':';
lw = 1.3;
mks = 'o';
mkz = 8;

ls2 = '--';
lw2 = 1.3;
mks2 = 'None';
mkz2 = 8;

figure(1);
hold off;
semilogy(Vdcs, abs(h1), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
hold on;
semilogy(Vdcs, abs(h2), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h3), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h4), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h5), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Power, b1/a2");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
legend("1st Harmonic", "2nd Harmonic", "3rd Harmonic", "4th Harmonic", "5th Harmoinc");

figure(2);
hold off;
plot(Vdcs, angle(h1).*180./pi, 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
hold on;
plot(Vdcs, angle(h2).*180./pi, 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Phase, b1/a2 (deg)");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
legend("1st Harmonic", "2nd Harmonic");

figure(3);
hold off;
semilogy(Vdcs, abs(h1), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
hold on;
semilogy(Vdcs, abs(h2), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h3), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h4), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h5), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(f1w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2);
semilogy(Vdcs, abs(f2w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2);
semilogy(Vdcs, abs(f3w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Power, b1/a2");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
legend("Fund. Meas", "2nd H. Meas.", "3rd H. Meas", "4th H. Meas", "5th H. Meas", "Fund. Predicted", "2nd H Pred.", "3rd H. Pred");

%% Run Fit Algorithm

qs = linspace(0.03, 0.031, 3);
rs = linspace(0.01, 0.5, 301);
L0s = logspace(-17, -13, 101);
targ = struct('h1', h1, 'h2', h2, 'h3', h3, 'h4', h4);

% Fit parameters to data
[q_opt, r_opt, L0_opt, best_err] = fit_harmoincs4(qs, rs, L0s, params, targ);

% Calculate resulting functions
params.q = q_opt;
params.r = r_opt;
params.L0 = L0_opt;
[f1w, f2w, f3w, f4w] = calc_harmonics(params);

%% Plot Optimized

c1 = [0, 0, .7];
c2 = [.7, 0, 0];
c3 = [0, .7, 0];
c4 = [.5, .5, .5];

mks = 'o';
mkz_a = ones(1, numel(h1)).*20;

figure(4);
hold off;
scatter(Vdcs, abs(h1), mkz_a, 'Marker', mks, 'MarkerEdgeColor', c1, 'MarkerFaceColor', c1);
set(gca,'yscale','log')
hold on;
scatter(Vdcs, abs(h2), mkz_a, 'Marker', mks, 'MarkerEdgeColor', c2, 'MarkerFaceColor', c2);
scatter(Vdcs, abs(h3), mkz_a, 'Marker', mks, 'MarkerEdgeColor', c3, 'MarkerFaceColor', c3);
scatter(Vdcs, abs(h4), mkz_a, 'Marker', mks, 'MarkerEdgeColor', c4, 'MarkerFaceColor', c4);
semilogy(Vdcs, abs(f1w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', c1);
semilogy(Vdcs, abs(f2w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', c2);
semilogy(Vdcs, abs(f3w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', c3);
semilogy(Vdcs, abs(f4w), 'LineStyle', ls2, 'LineWidth', lw2, 'Marker', mks2, 'MarkerSize', mkz2, 'Color', c4);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Power, b1/a2");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
legend("Fund. Meas", "2nd H. Meas.", "3rd H. Meas", "4th H. Meas", "Fund. Predicted", "2nd H Pred.", "3rd H. Pred", "4th. Pred");
ylim([10e-7, 1]);

%% Function Definitions

function [q, r, L0, err] = fit_harmoincs4(qs, rs, L0s, params, target)
	
	h1 = target.h1;
	h2 = target.h2;
	h3 = target.h3;
	h4 = target.h4;

	% Create local variables
	Idc = params.Idc;
	Ipp = params.Ipp;
	q = params.q;
	r = params.r;
	L0 = params.L0;
	f = params.f;
	R_power = params.R_power;
	
	w = 2.*3.14159.*f;
	dIdt = Ipp.*w.*sqrt(2)./2;

	coef = L0 .* dIdt;

	% Define tracking variables
	best_err = -1;
	best_q = -1;
	best_r = -1;
	best_L0 = -1;
	
	num_guesses = numel(qs)*numel(L0s)*numel(rs);
	displ("Number of guesses: ", num_guesses);

	% For displt
	nb = 0;
	
	% Brute force the fit
	idx = 0;
	idx_last = 0;
	for L0_g = L0s
		for q_g = qs
			for r_g = rs
			
				% Print update message
				if idx - idx_last > 500e3
					displ("Percent complete: ", num2fstr(idx/num_guesses*100), "%");
					idx_last = idx;
				end
				
				% Update parameters
				params.r = r_g;
				params.q = q_g;
				params.L0 = L0_g;

				% Calculate harmonics for guess
				[f1g, f2g, f3g, f4g] = calc_harmonics(params);

				% Calculate error
				err = sum(( abs(f2g) - abs(h2) ).^2);
				err = err + sum(( abs(f3g) - abs(h3) ).^2);
				err = err + sum(( abs(f4g) - abs(h4) ).^2);

				% Update best
				if err < best_err || best_err < 0
					displ("Best error changed to ", err*1000, " (Improvement: ", best_err-err ,") (q=", q_g, ", r=", r_g,")");
					best_err = err;
					best_q = q_g;
					best_L0 = L0_g;
					best_r = r_g;
				end

				idx = idx + 1;
			end
		end
	end
	
	q = best_q;
	r = best_r;
	L0 = best_L0;
	err = best_err;
	
	barprint("Optimization Finished");
	displ("q = ", q);
	displ("r = ", r);
	displ("L0 = ", L0);
	displ("error = ", err);
	barprint("");
	
	
	
end

function [f1w, f2w, f3w, f4w] = calc_harmonics(stats)

	% Create local variables
	Idc = stats.Idc;
	Ipp = stats.Ipp;
	q = stats.q;
	r = stats.r;
	L0 = stats.L0;
	f = stats.f;
	R_power = stats.R_power;
	
	w = 2.*3.14159.*f;
	dIdt = Ipp.*w.*sqrt(2)./2;

	coef = L0 .* dIdt;
	
	% Calculate products
	f1w = coef.*(2.*Idc.*Ipp./q.^2 + (4.*Idc.^3.*Ipp + 3.*Idc.*Ipp.^3)./r.^4);
	f2w = coef.*(Ipp.^2./2./q.^2 + (3.*Idc.^2.*Ipp.^2 + Ipp.^4./2)./r.^4);
	f3w = coef.*(Idc.*Ipp.^3./r.^4);
	f4w = coef.*(Ipp.^4./8);
	
	f1w = f1w.^2/R_power;
	f2w = f2w.^2/R_power;
	f3w = f3w.^2/R_power;
	f4w = f4w.^2/R_power;

end















