

%% Run Microsim Analysis

% Define sim conditions
freqs = 10e9;

% Define elements
loadElement = shuntRes(50, freqs);
cable = tlin(50, 45, freqs, true);
cable2 = tlin(50, 45, freqs, true);

% Input impedance looking into 50ohm connected to load
% net = copyh(cable);

theta = linspace(0, 360, 361);

net = copyh(loadElement);
net.series(cable);
zina = net.Zin_along(88.4, theta, NaN, true);

%% Plot Results

figure(1);
plot(theta, zina);
grid on;
xlim([0, 360]);
setxtick(45, false);