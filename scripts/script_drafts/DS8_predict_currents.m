freqs_length = linspace(1, 360, 301);

% Define length
theta_cable = freqs_length;
theta_chip  = freqs_length;

% Define elements
load = shuntRes(50, freqs_length);
cable = tlin(50, theta_cable, freqs_length, true);
chip =  tlin(90, theta_chip, freqs_length, true);

% Input impedance looking into 50ohm connected to load
% net = copyh(load);
% net.series(cable);
net = copyh(cable);
% net.series(load);
zin_A = net.Zin(50);
S_A = net.S();
S11_A = S_A(1,1,:);
S11_A = permute(S11_A, [1, 3, 2]); % Reshape S11
displ("Zin into load cable: ", num2fstr(zin_A(1)), " ohms");

% Input ipmedance looking into the chip from generator side
net.series(chip);
zin_B = net.Zin(50);
displ("Zin into Chip: ", num2fstr(zin_B(1)), " ohms");

% Input ipmedance looking into system
net.series(cable);
zin_C = net.Zin(50);
displ("Zin into system: ", num2fstr(zin_C(1)), " ohms");

figure(1);

subplot(3, 1, 1);
hold off
plot(freqs_length, abs(zin_A));
xlim([0, max(freqs_length)]);
grid on;
title("Load Cable Input Impedance");
ylabel("Impedance (Ohms)");
xlabel("Cable length (deg)");
set(gca,'Xtick',0:45:freqs_length(end))
force0y;

subplot(3, 1, 2);
hold off
plot(freqs_length, abs(zin_B));
xlim([0, max(freqs_length)]);
grid on;
title("Chip Input Impedance");
ylabel("Impedance (Ohms)");
xlabel("Cable length (deg)");
set(gca,'Xtick',0:45:freqs_length(end))

subplot(3, 1, 3);
hold off
plot(freqs_length, abs(zin_C));
xlim([0, max(freqs_length)]);
grid on;
title("System Input Impedance");
ylabel("Impedance (Ohms)");
xlabel("Cable length (deg)");
set(gca,'Xtick',0:45:freqs_length(end))