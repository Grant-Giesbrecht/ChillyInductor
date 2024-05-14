DATAPATH = fullfile('/Volumes/M6 T7S/ARC0 PhD Data/RP-21 Kinetic Inductance 2023/Data/group4_extflash/NIST September data', "TDR");
load(fullfile(DATAPATH, "TDR_Dataset_1.mat"));

figure(1);
hold off;

figure(2);
hold off;

interest_region = struct('x', [0.09, 0.14], 'y', [56, 61]); 

tdr_Z = 305; % Impedance of DC path
Vp0 = 86.207e6; % m/s
len = 0.5;

% Scan over points
mksz = 100;
CM = resamplecmap('parula', numel(tdr_data)+1);
cmd = brighten(colormap('parula'), -0.5);
CMd = resamplecmap(cmd, numel(tdr_data)+1);
idx = 0;
t_max = zeros(1, numel(tdr_data));
handles = gobjects(1, numel(tdr_data)*3);
horder = [];
warning off;
for ds = tdr_data
    idx = idx + 1;
    
    figure(1);
    plot(ds.t./1e-9, ds.R, "DisplayName", "V_{DC} = "+num2str(ds.Vdc), 'Color', CM(idx, :));
    hold on;
	
	% Mask data region to fit
    mask = (ds.t >= interest_region.x(1)*1e-9 ) & (ds.t <= interest_region.x(2)*1e-9);
    t_reg = ds.t(mask);
    R_reg = ds.R(mask);
	
	% Apply fit
	pf = polyfit(t_reg,R_reg,6);	
	ds.sdR = polyval(pf,t_reg);
    ds.sdt = t_reg;
	
	% Identify max point
	[mv, I] = max(ds.sdR);
	t_max(idx) = t_reg(I);
	
	figure(2);
    handles(idx*3-2) = plot(ds.sdt./1e-9, R_reg, "DisplayName", "V_{DC} = "+num2str(ds.Vdc), 'Color', CM(idx, :), 'Marker', '.', 'LineWidth', 0.01);
    hold on;
    handles(idx*3-1) = plot(ds.sdt./1e-9, ds.sdR, "DisplayName", "Smoothed V_{DC} = "+num2str(ds.Vdc), 'Color', CMd(idx, :), 'LineStyle', '--', "LineWidth", 4);
	handles(idx*3) = scatter(ds.sdt(I)./1e-9, ds.sdR(I), mksz(1), 'Marker', 'o', 'MarkerFaceColor', CMd(idx, :), 'MarkerEdgeColor', CM(idx, :), 'DisplayName', "Peak: V_{DC} = "+num2str(ds.Vdc));
end
warning on;

h_order = 1:3:numel(tdr_data)*3-2;
h_order = cat(2, h_order, 2:3:numel(tdr_data)*3-1);
h_order = cat(2, h_order, 3:3:numel(tdr_data)*3);

figure(1);
grid on;
xlabel("Time (ns)");
ylabel("Characteristic Impedance (\Omega)");
title("TDR Datatset 1, Chip 2");
legend('NumColumns', 2);
rectangle('Position', [interest_region.x(1), interest_region.y(1), interest_region.x(2)-interest_region.x(1), interest_region.y(2)-interest_region.y(1)], 'LineWidth', 1);

figure(2);
grid on;
xlabel("Time (ns)");
ylabel("Characteristic Impedance (\Omega)");
title("TDR Datatset 1, Chip 2");
legend(handles([h_order]), 'NumColumns', 3)
xlim(interest_region.x);
ylim(interest_region.y);

%% Find q from dt and Vdc

dT = t_max - t_max(1);
Vdc = [tdr_data.Vdc];

I = Vdc./305;

% Calculate q
q = sqrt(len./abs(dT(2:end))./Vp0).*abs(I(2:end));

figure(3);
plot(Vdc(2:end), q.*1e3, 'LineStyle', ':', 'Marker', 'o', 'MarkerSize', 12, 'MarkerFaceColor', [0, 0.6, 0]);
xlabel("V_{DC} (V)");
ylabel("q (mA)");
title("q estimate from TDR");
grid on;

displ("mean q: ", mean(q).*1e3, " mA");
displ("mean q(2:end): ", mean(q(2:end)).*1e3, " mA");

displ("stdev q: ", std(q).*1e3, ' mA');
displ("stdev q(2:end): ", std(q(2:end)).*1e3, ' mA');

return;

%% Format for publication

figure(1);
set(gca,'FontSize', 22, 'FontName', 'Times New Roman');
ylabel("Impedance (\Omega)");
legend('location', 'best');
title("Zoomed-In TDR Feature");
figure(2);
set(gca,'FontSize', 22, 'FontName', 'Times New Roman');
figure(3);
set(gca,'FontSize', 22, 'FontName', 'Times New Roman');
pbaspect([1.0000    0.3825    0.3825]);
title("Estimate of q from TDR");

























