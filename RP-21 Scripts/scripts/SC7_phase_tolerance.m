%% SC7 Phase Tolerance
%
% Previously named: DS12_phase_tolerance.m
%
% Used to show that even a 1% error in Vp or chip length can add an entire
% wavelength wrap, meaning that unless we have a crazy precise measurement
% (less than 1% error), we essentially know nothing about the phase
% remainder.

%% Nominal Case

vp_tol = 1;
len_tol = 1;

num_lambda = 0.5/f2l(10e9, 0.2873); % Vp from Nathan's spreadsheet, calc'd from TDR

num_lambdap = 0.5/f2l(10e9, 0.2873*(100+vp_tol)/100);
num_lambdam = 0.5/f2l(10e9, 0.2873*(100-vp_tol)/100);

num_lambda_lp = (0.5*(100+len_tol)/100)/f2l(10e9, 0.2873);
num_lambda_lm = (0.5*(100-len_tol)/100)/f2l(10e9, 0.2873);

displ("Nominal Case");
displ("  Expected Phase: ", WLstr(num_lambda), " deg");
displ("Phase Range:");
displ("    Vp (+/- ", vp_tol ,"%):  ", WLstr(num_lambdap), " deg");
displ("                  ", WLstr(num_lambdam), " deg");
displ("    Len (+/- ", vp_tol ,"%): ", WLstr(num_lambda_lm), " deg");
displ("                  ", WLstr(num_lambda_lp), " deg");

function S = WLstr(num_lambda)
	% Show the wavelength as a string
	
	phase = mod(num_lambda*360, 360);
	S = strcat("[", num2str(floor(num_lambda)), "x] + ", num2str(phase));
end