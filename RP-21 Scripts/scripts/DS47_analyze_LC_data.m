
% Read data file
net=sparameters(fullfile("script_assets", "SC20", "sweep4_0V.s2p"));

% Extract S-parameters
freq = net.Frequencies;
freq_GHz = freq./1e9;
S11_lin = flatten(net.Parameters(1, 1, :));
S11 = lin2dB(abs(S11_lin));
S21_lin = flatten(net.Parameters(1, 2, :));
S21 = lin2dB(abs(S21_lin));

% Plot raw data
figure(1);
NR = 1;
NC = 2;
subplot(NR, NC, 1);
hold off;
plotsc(S11);
subplot(NR, NC, 2);
hold off;
plot(freq_GHz, S11, 'LineStyle', '-', 'Color', [0.7, 0, 0]);
hold on;
plot(freq_GHz, S21, 'LineStyle', '-', 'Color', [0, 0, 0.7]);
grid on;
xlabel("Frequency (GHz)");
ylabel("S-Parameters (dB)");
legend("S_{11}", "S_{21}");


