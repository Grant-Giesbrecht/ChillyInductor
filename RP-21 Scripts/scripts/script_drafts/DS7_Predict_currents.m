G1 = Z2G(90, 50);
G2 = Z2G(50, 90);

theta = 2.*pi.*linspace(54, 58, 301);

G_tot = G1 + G2.*exp(-2.*1i.*theta);
T_tot = 1 + G_tot;

figure(1);
hold off;
plot(theta./2./pi, abs(T_tot));
hold on;
plot(theta./2./pi, abs(G_tot));
grid on;
legend("T", "\Gamma");
xlabel("Chip length (\lambda)");
ylabel("(Unitless)");
title("\Gamma vs T");

figure(2);
subplot(2, 1, 1);
hold off;
plot(theta./2./pi, real(G_tot));
hold on;
plot(theta./2./pi, imag(G_tot));
grid on;
legend("Re{\Gamma}", "Im{\Gamma}");
xlabel("Chip length (\lambda)");
ylabel("Reflection Coefficient");
title("\Gamma Components");

subplot(2, 1, 2);
hold off;
plot(theta./2./pi, real(T_tot));
hold on;
plot(theta./2./pi, imag(T_tot));
grid on;
legend("Re{T}", "Im{T}");
xlabel("Chip length (\lambda)");
ylabel("Transmission Coefficient");
title("T Components");