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

ls = ':';
lw = 1.3;
mks = 'o';
mkz = 8;

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
% plot(Vdcs, angle(h3).*180./pi, 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
% plot(Vdcs, angle(h4).*180./pi, 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
% plot(Vdcs, angle(h5).*180./pi, 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Phase, b1/a2 (deg)");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
legend("1st Harmonic", "2nd Harmonic");

%% Fit Data to Mixing Product Equations

wa = 2.*pi.*10; % 10 GHz from SG
Rsystem = 100; % Impedance of system (ohms)

Idc = Vdcs./Rsystem;

stats.h1 = h1;
stats.h2 = h2;
stats.h3 = h3;
stats.wa = wa;
stats.Idc = Idc;

% Input amplitude of signal
% as = [40e-6, 1e-3, 1];
as = logspace(-4, -5, 1000);
L0s = logspace(-4, 3, 1000);

[a, L0, err] = fit_mixing_products(flip(as), (L0s), stats);

displ(newline, "Fit Results:");
displ("     a: ", a);
displ("    L0: ", L0);
displ(" Error: ", err);

%% Plot Best Fit

[h1g, h2g, h3g] = calc_mixing_products(a, L0, stats);

figure(3);
hold off;
semilogy(Vdcs, abs(h1), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
hold on;
semilogy(Vdcs, abs(h2), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h3), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
% semilogy(Vdcs, abs(h4), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
% semilogy(Vdcs, abs(h5), 'LineStyle', ls, 'LineWidth', lw, 'Marker', mks, 'MarkerSize', mkz);
semilogy(Vdcs, abs(h1g), 'Color', [0, 0, .8]);
semilogy(Vdcs, abs(h2g), 'Color', [.8, 0, .8]);
% semilogy(Vdcs, abs(h3g), 'Color', [.6, .6, 0]);
xlabel("Bias Voltage (V)");
ylabel("Harmonic Power, b1/a2");
title("Harmonic Measurement, -10 dBm, 10 GHz");
grid on;
% legend("1st Harmonic", "2nd Harmonic", "3rd Harmonic", "4th Harmonic", "5th Harmoinc");













%% Functions

function [a, L0, err] = fit_mixing_products(as, L0s, stats)
	
	wa = stats.wa;
	Idc = stats.Idc;
	h1 = stats.h1;
	h2 = stats.h2;
	h3 = stats.h3;

	% Define tracking variables
	best_err = -1;
	best_a = -1;
	best_L0 = -1;
	
	num_guesses = numel(as)*numel(L0s);
	displ("Number of guesses: ", num_guesses);

	% Brute force the fit
	idx = 0;
	idx_last = 0;
	for a_g = as
		for L0_g = L0s
			
			if idx - idx_last > 500e3
				displ("Percent complete: ", num2fstr(idx/num_guesses*100), " %");
				idx_last = idx;
			end

			% Calculate harmonics for guess
% 			h1g = L0_g.*a_g.*wa + Idc.^2.*a_g.*wa + a_g.^3.*wa./4;
% 			h2g = Idc.*a_g.^2.*wa;
% 			h3g = a_g.^3.*wa./4;
			[h1g, h2g, h3g] = calc_mixing_products(a_g, L0_g, stats);

			% Calculate error
% 			err = sum(( abs(h1g) - abs(h1) ).^2) + sum(( abs(h2g) - abs(h2) ).^2) + sum(( abs(h3g) - abs(h3) ).^2);
			err = sum(( abs(h2g) - abs(h2) ).^2);

			% Update best
			if err < best_err || best_err < 0
				displ("Best error changed to ", err*1000, " (Improvement: ", best_err-err ,") (a=", a_g, ", L0=", L0_g,")");
				best_err = err;
				best_a = a_g;
				best_L0 = L0_g;
			end
			
			idx = idx + 1;
		end
	end
	
	a = best_a;
	L0 = best_L0;
	err = best_err;
end

function [h1x, h2x, h3x] = calc_mixing_products(a_g, L0_g, stats)

	wa = stats.wa;
	Idc = stats.Idc;
	h1 = stats.h1;
	h2 = stats.h2;
	h3 = stats.h3;

	h1x = L0_g.*a_g.*wa + Idc.^2.*a_g.*wa + a_g.^3.*wa./4;
	h2x = Idc.*a_g.^2.*wa;
	h3x = a_g.^3.*wa./4;

end















