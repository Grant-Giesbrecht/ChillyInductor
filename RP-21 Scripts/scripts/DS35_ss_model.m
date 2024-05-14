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

pi = 3.1415926535;

%% Basic Inductor Voltage Model

I0 = 3e-3;
omega = 10e9*2*pi;

Ibias_A = Vdcs.*iv_conv;

part3_coef = -L0.*omega.*I0.^3./4./q.^2;

V_f1 = L0.*I0.*omega.*( 1 + abs(Ibias_A).^2./q.^2 + I0.^2./2./q.^2 ) + part3_coef;
V_f2 = L0.*omega.*I0.^2.*abs(Ibias_A)./q.^2;
V_f3 = abs(part3_coef);



%% Get measured data

load(dataset_path("DS5_FinePower_PO-1.mat"));

iv_conv = 9.5e-3; % A/V
c = struct('SG_power', -10);
c.Vnorm = 2e-3;
c.SG_power = -6;

% Calculate harmonics over bias sweep
[harms, norm, Vdcs] = getHarmonicSweep(ld, c);
fund = harms.h1;
h2 = harms.h2;
Ibias = Vdcs.*iv_conv;

lw = 1.5;
mks = 10;

figure(1);
hold off;
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h1)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0, .7], 'MarkerSize', mks);
hold on;
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h2)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
plot(Vdcs.*iv_conv.*1e3, VNA2dBm(abs(harms.h3)), 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0.7, 0, 0], 'MarkerSize', mks);
grid on;
legend("Fundamental", "2nd Harmonic", "3rd Harmonic");
xlabel("DC Bias Current (mA)");
ylabel("Harmonic Power (dBm)");

conv = 1e-9

plot(Ibias_A.*1e3, cvrt(V_f1.^2.*conv, 'W', 'dBm'), 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0, .5], 'MarkerSize', mks);
hold on;
plot(Ibias_A.*1e3, cvrt(V_f2.^2.*conv, 'W', 'dBm'), 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0, 0.5, 0], 'MarkerSize', mks);
plot(Ibias_A.*1e3, cvrt(V_f3.^2.*conv, 'W', 'dBm'), 'LineStyle', ':', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.5, 0, 0], 'MarkerSize', mks);
grid on;
legend("Meas. Fundamental", " Meas. 2nd Harmonic", " Meas. 3rd Harmonic", "Sim. Fundamental", "Sim. 2nd Harmonic", "Sim. 3rd Harmonic");
xlabel("DC Bias Current (mA)");
ylabel("Power (dBm)");

title("Crude Model vs Measurement (scale coef="+num2str(conv)+")");

% figure(2);
% hold off;
% semilogy(Ibias_A.*1e3, V_f1, 'LineStyle', ':', 'LineWidth', lw, 'Marker', '+', 'Color', [0, 0, .7], 'MarkerSize', mks);
% hold on;
% semilogy(Ibias_A.*1e3, V_f2, 'LineStyle', '--', 'LineWidth', lw, 'Marker', '*', 'Color', [0, 0.7, 0], 'MarkerSize', mks);
% semilogy(Ibias_A.*1e3, V_f3, 'LineStyle', '-.', 'LineWidth', lw, 'Marker', 'o', 'Color', [0.7, 0, 0], 'MarkerSize', mks);
% grid on;
% legend("Fundamental", "2nd Harmonic", "3rd Harmonic");
% xlabel("DC Bias Current (mA)");
% ylabel("Voltage (V)");




































