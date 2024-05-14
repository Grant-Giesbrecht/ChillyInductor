%% SC15
%
% Previously named: DS22_AWR_power_conservation
%
% Looks at total power from AWR simulation to prove that power is conserved. Limited by crappy source power meas

%% Load data from AWR
load('C:\Users\Grant Giesbrecht\OneDrive - UCB-O365\NIST\backup_Nonlin Sim 10GHz AllComponents.AP_HB.mat')

% COnvert to units that aren't innane
P_fund_W = cvrt(P_fund, 'dBW', 'W');
P_2H_W = cvrt(P_2H, 'dBW', 'W');
P_3H_W = cvrt(P_3H, 'dBW', 'W');
P_4H_W = cvrt(P_4H, 'dBW', 'W');
P_5H_W = cvrt(P_5H, 'dBW', 'W');
P_src_W = cvrt(P_src, 'dBW', 'W');
P_tot_W = P_fund_W + P_2H_W + P_3H_W + P_4H_W + P_5H_W;

% Convert to units that aren't innane
P_fund_dBm = cvrt(P_fund, 'dBW', 'dBm');
P_2H_dBm = cvrt(P_2H, 'dBW', 'dBm');
P_3H_dBm = cvrt(P_3H, 'dBW', 'dBm');
P_4H_dBm = cvrt(P_4H, 'dBW', 'dBm');
P_5H_dBm = cvrt(P_5H, 'dBW', 'dBm');
P_src_dBm = cvrt(P_src, 'dBW', 'dBm');
P_tot_dBm = cvrt(P_tot_W, 'W', 'dBm');

c1 = [.7, 0, 0];
c2 = [0, 0, .7];
c3 = [0, .6, .2];
c4 = [.6, .6, 0];
c5 = [0.3, .1, 0.1];
c6 = [0.7, 0.7, 0.7];
c7 = [0.5, 0.8, 0];

figure(1);
hold off;
plot(Ibias, P_fund_dBm, 'LineStyle', ':', 'Marker', 'o', 'Color', c1);
hold on;
plot(Ibias, P_2H_dBm, 'LineStyle', ':', 'Marker', 'o', 'Color', c2);
plot(Ibias, P_3H_dBm, 'LineStyle', ':', 'Marker', 'o', 'Color', c3);
plot(Ibias, P_4H_dBm, 'LineStyle', ':', 'Marker', 'o', 'Color', c4);
plot(Ibias, P_5H_dBm, 'LineStyle', ':', 'Marker', 'o', 'Color', c5);
plot(Ibias, P_tot_dBm, 'LineStyle', '--', 'Marker', '+', 'Color', c6);
grid on;
xlabel("Bias Current (mA)");
ylabel("Power (dBm)");
title("AWR 10 GHz Nonlinear Simulation Results");
legend("Fundamental", "2nd Harm.", "3rd Harm.", "4th Harm.", "5th Harm.", 'Total Power');
ylim([-60, 10]);

figure(2);
hold off;
plot(Ibias, P_fund_W.*1e3, 'LineStyle', ':', 'Marker', 'o', 'Color', c1);
hold on;
plot(Ibias, P_2H_W.*1e3, 'LineStyle', ':', 'Marker', 'o', 'Color', c2);
plot(Ibias, P_3H_W.*1e3, 'LineStyle', ':', 'Marker', 'o', 'Color', c3);
plot(Ibias, P_4H_W.*1e3, 'LineStyle', ':', 'Marker', 'o', 'Color', c4);
plot(Ibias, P_5H_W.*1e3, 'LineStyle', ':', 'Marker', 'o', 'Color', c5);
plot(Ibias, P_tot_W.*1e3, 'LineStyle', '--', 'Marker', '+', 'Color', c6);
plot(Ibias, P_src_W.*1e3, 'LineStyle', '--', 'Marker', '+', 'Color', c7);
grid on;
xlabel("Bias Current (mA)");
ylabel("Power (mW)");
title("AWR 10 GHz Nonlinear Simulation Results");
legend("Fundamental", "2nd Harm.", "3rd Harm.", "4th Harm.", "5th Harm.", 'Total Power', 'Source power');