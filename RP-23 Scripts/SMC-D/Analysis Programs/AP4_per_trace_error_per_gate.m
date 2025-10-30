%% Step 1

% Remap to |0> and |1>
ampl = mean(X0All - X1All);
offset = mean(X1All);

data = (data_raw - offset)/ampl;

% Assuming sequence (1x25) and data (15x25)
nRepeats = size(data, 1);
nPoints  = size(data, 2);

% Preallocate
fitParams = zeros(nRepeats, 3); % [A, tau, B]
epg = zeros(nRepeats, 1);
RMSEs = zeros(nRepeats, 1);
RSquares = zeros(nRepeats, 1);

for i = 1:nRepeats
	
	x = sequence(:);
    y = data(i, :).';
	
	%=============  From XLD  ================
	[xData, yData] = prepareCurveData( sequence, yfit2 );
	ft = fittype( '(a*b.^x+c)', 'independent', 'x', 'dependent', 'y' );
	opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
	opts.Display = 'Off';
	opts.StartPoint = [0.5 0.99 0.5];
	[fitresult, gof] = fit( x, y, ft, opts );
	
	RMSEs(i) = gof.rmse;
	RSquares(i) = gof.rsquare;
	
	if abs(fitresult.a) > 2 || abs(fitresult.c) > 2
		fitParams(i, :) = [nan nan nan];
	else
		fitParams(i, :) = [fitresult.a, fitresult.b, fitresult.c];
	end
	
	epg(i) = 1 / 2 * (1 - fitresult.b);
	
% 	%========== From the bats ==============
% 	
% 
%     
%     % Define the exponential model
%     f = fittype('A*exp(-x/tau) + B', 'independent', 'x', 'coefficients', {'A', 'tau', 'B'});
%     
%     % Initial guesses
%     A0 = y(1) - y(end);
%     tau0 = mean(sequence);
%     B0 = y(end);
%     
%     % Perform the fit
%     fitObj = fit(x, y, f, 'Start', [A0, tau0, B0]);
%     
%     % Save parameters
%     fitParams(i, :) = [fitObj.A, fitObj.tau, fitObj.B];
%     
%     % Compute "error per gate" metric — common in RB: p = (1 - EPG)
%     % Here the exponential argument gives the decay per gate:
%     epg(i) = 1 - exp(-1 / fitObj.tau);
end


%% Step 2

%Remove nan rows
rowsToRemove = any(isnan(fitParams), 2);  % True for rows that are all NaN
fitParams(rowsToRemove, :) = [];          % Remove those rows


epg_mean = mean(epg);
epg_std  = std(epg);

fprintf('Mean error per gate: %.4g ± %.4g\n', epg_mean, epg_std);

figure; hold on;
xFit = linspace(min(sequence), max(sequence), 200);

for i = 1:size(fitParams, 1)
% 	(a*b.^x+c)
	yFit = fitParams(i,1)*fitParams(i,2).^xFit+fitParams(i,3);
    plot(xFit, yFit, 'Color', [0.6 0.6 0.6]);
end

% for i = 1:nRepeats
%     yFit = fitParams(i,1)*exp(-xFit/fitParams(i,2)) + fitParams(i,3);
%     plot(xFit, yFit, 'Color', [0.6 0.6 0.6]);
% end

% Average fit using mean parameters
meanFit = mean(fitParams, 1);
plot(xFit, meanFit(1)*meanFit(2).^xFit+meanFit(3), 'r', 'LineWidth', 2);

xlabel('Sequence Length');
ylabel('Visibility');
title(sprintf('(Mean EPG = %.3g ± %.3g)', epg_mean, epg_std));
grid on;

