% Prev. DS30
%
% This script numerically checks Pozar eq. 2.71 vs 2.70 and shows they are
% only equivalent when gammaL or gammaG is zero. If a large mismatch
% occurs, they can disagree by a lot! In realistic scenarios (50 ohm gen to
% 10 ohm load), it can be as much as 60 percent!


Vg = 1;

Zg = 50;
Z0 = 10:0.1:50;
ZL = 10;
% BL = 0:.1:pi;
BL = pi/4;
Zin = xfmr2zin(Z0, ZL, BL);
gammaL = Z2G(ZL, Z0);
gammaG = Z2G(Z0, Zg);

j = complex(0,1);

V0p_f1 = Vg.*Zin./(Zin + Zg)./(exp(j.*BL) + gammaL.*exp(-j.*BL));

V0p_f2 = Vg.*Z0./(Z0 + Zg).*exp(-j.*BL)./(1-gammaL.*gammaG.*exp(-2.*j.*BL));

error = 100.*abs(V0p_f1 - V0p_f2)./min([abs(V0p_f1), abs(V0p_f2)]);

displ("Form 1: ", V0p_f1);
displ("Form 2: ", V0p_f2);

displ("|Form 1|: ", abs(V0p_f1));
displ("|Form 2|: ", abs(V0p_f2));

displ("Error (%): ", error);


figure(1);
hold off;
subplot(1, 2, 1);
plot(error, 'LineStyle', ':', 'Marker', '+');
xlabel("Sweep Index");
ylabel("Error (%)");
title("Error Analysis");
grid on;

subplot(1,2,2);
hold off;
plot(real(V0p_f1), 'LineStyle', ':', 'Color', [0, 0, 0.7]);
hold on;
plot(real(V0p_f2), 'LineStyle', '--', 'Color', [0, 0, .9]);

plot(imag(V0p_f1), 'LineStyle', ':', 'Color', [0.6, 0, 0]);
plot(imag(V0p_f2), 'LineStyle', '--', 'Color', [0.9, 0, 0]);

xlabel("Sweep Index");
ylabel("Value");
title("Value Comparison");
legend("F1 - Real", "F2 - Real", "F1 - Imag", "F2 - Imag");
grid on;

