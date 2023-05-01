N = 50;

% freqs = zeros(1, N) + 10e9;
freqs = logspace(9, 11, N);
Zterm = 100;

VF_chip = 0.303;
VF_cable = 0.7;

len_chip  = zeros(1, N) + 0.5;
len_cable = zeros(1, N) + 0.25;

theta_1 = 2.*pi.*len_cable./f2l(freqs, VF_cable);
theta_2 = 2.*pi.*len_chip./f2l(freqs, VF_chip);

% theta1 = 2*pi*6.3;
% theta2 = 2*pi*12;
% theta3 = theta1;

% Initialize with termination
cable = tlin(50, theta1, freqs);
chip = tlin(90, theta2, freqs);

net = copyh(cable);
net.series(chip);
net.series(cable);

ZI = net.Zin(Zterm);

% displ("Network input Impedace: ")
% displ("  Zin = ", net.Zin(Zterm), " Ohms");

figure(1);
hold off;
semilogx(freqs, (real(ZI)));
hold on;
semilogx(freqs, (imag(ZI)));