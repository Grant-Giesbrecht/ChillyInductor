% Assume you already have:
% sequence -> 1x25 array of x positions
% data -> 15x25 array of y values

% Repeat the x values 15 times to match the size of data
x = repmat(sequence, size(data_raw, 1), 1);

% Remap to |0> and |1>
ampl = mean(X0All - X1All);
offset = mean(X1All);

data_mapped = (data_raw - offset)/ampl;

% Compute mean and std for each x
meanVals = mean(data_mapped, 1);
stdVals  = std(data_mapped, 0, 1);

% Flatten both x and data so they can be plotted as vectors
x = x(:);
y = data_mapped(:);

% Make the scatter plot
figure(1);
hold off;
% scatter(x, y, 'Marker', '.');
scatter(x, y, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
errorbar(sequence, meanVals, stdVals, 'k', 'LineWidth', 1.5, 'CapSize', 6);
plot(sequence, meanVals, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 15);

fit_x_vals = linspace(0, max(sequence), 101);
plot(fit_x_vals, fitresult(fit_x_vals), 'LineStyle', '-', 'LineWidth', 0.5, 'Color', [0.75, 0.2, 0.2]);

% Label axes
xlabel('Sequence Length');
ylabel('Visibility');
title('Scatter plot of repeated measurements');
grid on;
