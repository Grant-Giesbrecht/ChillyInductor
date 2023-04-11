%% Import Data

% Data File
datafolder = '';
filename = "PO1.mat";

% Import data
load(fullfile(datafolder, filename));
ds = ld;

% Create conditoins
c = struct('SG_power', -10, 'Vdc', 0.25);
c.harms = unique([ld.harmonic]);
c.Vnorm = 2e-3;

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

%% Plot Data

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

%% Find 'q'

