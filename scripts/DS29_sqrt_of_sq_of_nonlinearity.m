t = linspace(0, 10, 301);

Idc = 1;
Iac = 2;
I = Idc + Iac.*sin(t.*2.*3.14);

Isq = I.^2;
I0 = sqrt(Isq);

figure(1);
hold off;
plot(t, I, 'Color', [0.7, 0, 0], 'LineStyle', '--', 'LineWidth', 1.5);
hold on;
plot(t, Isq, 'Color', [0, 0.7, 0], 'LineStyle', '-.', 'LineWidth', 1.5);
plot(t, I0, 'Color', [0, 0, 0.7], 'LineStyle', ':', 'LineWidth', 1.5);
legend("I", "I^2", "sqrt(I^2)");
grid on;
xlabel("Time (a.u.)");
ylabel("I (a.u)")