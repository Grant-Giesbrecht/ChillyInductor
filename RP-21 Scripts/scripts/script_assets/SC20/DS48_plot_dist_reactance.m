ddf = DDFIO();
ddf.load(fullfile("script_assets", "SC20", "distributed_reactance.ddf"));
ddf.assignAll();

mkz = 8;
lw = 1.5;

figure(1);
plot(V, C_pF, 'LineStyle', ':', 'Marker', 'o', 'Color', [0, 0, 0.6], 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Bias Voltage (V)");
ylabel("Distributed Capacitance (pF/m)");
title("Distributed Capacitance over Bias");
ylim([148, 153]);
yticks(146:1:155);

figure(2);
plot(V, L_nH, 'LineStyle', ':', 'Marker', 'o', 'Color', [0, 0.6, 0], 'LineWidth', lw, 'MarkerSize', mkz);
grid on;
xlabel("Bias Voltage (V)");
ylabel("Distributed Inductance (nH/m)");
title("Distributed Inductance over Bias");

