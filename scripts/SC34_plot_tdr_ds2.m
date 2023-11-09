DATAPATH = fullfile("/", "Volumes", "NO NAME", "NIST September data", "TDR");
load(fullfile(DATAPATH, "TDR_Dataset_2.mat"));

warning("X axis units wrong. For some reason I had to add an extra 1e3!");

figure(4);
hold off;

% 34635 34825
interest_region = struct('x', [34635/1e3, 34825/1e3], 'y', [48, 62]); 

tdr_Z = 305; % Impedance of DC path
Vp0 = 86.207e6; % m/s
len = 0.5;

% Scan over points
mksz = 100;
cmd = brighten(colormap('parula'), -0.5);
CMd = resamplecmap(cmd, numel(tdr_data)+1);
idx = 0;
t_max = zeros(1, numel(tdr_data));
handles = gobjects(1, numel(tdr_data)*3);
horder = []
for ds = tdr_data
    idx = idx + 1;
    
    figure(4);
    plot(ds.t./1e-9./1e3, ds.R, "DisplayName", "V_{DC} = "+num2str(ds.Vdc), 'Color', CM(idx, :), 'LineWidth', 1.5);
    hold on;
	
	break
end

h_order = 1:3:numel(tdr_data)*3-2;
h_order = cat(2, h_order, 2:3:numel(tdr_data)*3-1);
h_order = cat(2, h_order, 3:3:numel(tdr_data)*3);

figure(4);
grid on;
xlabel("Time (ns)");
ylabel("Characteristic Impedance (\Omega)");
title("TDR Datatset 1, Chip 2");
legend();
rectangle('Position', [interest_region.x(1), interest_region.y(1), interest_region.x(2)-interest_region.x(1), interest_region.y(2)-interest_region.y(1)], 'LineWidth', 1);























