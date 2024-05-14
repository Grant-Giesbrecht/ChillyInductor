%% SC13
%
% Previously named DS20_fundamental_change_with_bias.m
%
% Looks at impact of including feedback (both on VF and I) (over freq)
%
% Looks at how the nonlinearity of the inductor impacts phase. Build off of
% SC11.

%% Set Conditions

% Simulation variables
freqs = linspace(10e9, 11e9, 301);

% VF variables
L0 = 1e-6;
C_ = 121e-12;
q = 0.190;

% System Variables
chip_len = 0.5;
Z0_chip = 88.4;
Z0_cable = 50;
P_dBm = 6;
P_watts = cvrt(P_dBm, 'dBm', 'W');
Vgen = sqrt(P_watts*200);
Zsrc = 50; % Generator source impedance

% Optimization parameters
Iac_tol = 1e-5; % [A] tolerance for convergence
conv_coef = 0.5; % Coefficient multiplying error for correction

%% Run Initial Simulation, Not Considering Feedback-Mechanism

VF_guess = 1/sqrt(L0*C_)/3e8;

% Define length (degrees)
theta_cable = 45;
theta_chip  = 360.*chip_len./f2l(freqs, VF_guess);

% Define elements
load_elmt = shuntRes(50, freqs);
cable = tlin(50, theta_cable, freqs, true);
cable2 = tlin(50, theta_cable, freqs, true);
chip =  tlin(Z0_chip, theta_chip, freqs, true);

% Input impedance looking into 50ohm connected to load
net = copyh(load_elmt);
net.series(cable);
zin_A = net.Zin();

% Input ipmedance looking into the chip from generator side
net.series(chip);
zin_B = net.Zin();

% Input ipmedance looking into system
net.series(cable2);
zin_C = net.Zin();

% Find Currents, Voltages, Input Impedances, Powers, etc
P0 = 1./2.*Vgen.^2.*real(zin_C)./( (real(zin_C) + real(Zsrc)).^2 + (imag(zin_C) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
Iac_initial_guess = sqrt(P0./50);

%% Solve for expected currents

% Constants for 'delta' parameter
NON_INIT = 0;
INCREASE = 1;
DECREASE = -1;

% Initialize solution array
Iac_exp = zeros(1, numel(freqs));
VF_exp = zeros(1, numel(freqs));

% Loop over each frequency
idx = 0;
for f_loop = freqs
	idx = idx + 1;
	
	% Get initial guess for current (From Microsim result without feedback)
	Iac_loop = Iac_initial_guess(idx);
	
	% Reset convergence variables
	 conv_fac = conv_coef;
	 delta = NON_INIT;
	
	% Loop until solution converges
	while true
		
		% Estimate VF
		Vp = 1/sqrt(L0*C_)*(1-Iac_loop^2/q^2);
		VF = Vp/3e8;
		
		%---------------------- Run Microsim analysis -------------------------
		
		% Define length (degrees)
		theta_cable = 45;
		theta_chip  = 360.*chip_len./f2l(f_loop, VF);
		
		% Define elements
		load_elmt = shuntRes(50, f_loop);
		cable = tlin(50, theta_cable, f_loop, true);
		cable2 = tlin(50, theta_cable, f_loop, true);
		chip =  tlin(Z0_chip, theta_chip, f_loop, true);
		
		% Input impedance looking into 50ohm connected to load
		net = copyh(load_elmt);
		net.series(cable);
		zin_A = net.Zin();
		
		% Input ipmedance looking into the chip from generator side
		net.series(chip);
		zin_B = net.Zin();
		
		% Input ipmedance looking into system
		net.series(cable2);
		zin_C = net.Zin();
		
		% Calculate expect power and current
		P0 = 1./2.*Vgen.^2.*real(zin_C)./( (real(zin_C) + real(Zsrc)).^2 + (imag(zin_C) + imag(Zsrc)).^2 ); % From Pozar_4e, page 77, eq. 2,76
		Iac_loop_f = sqrt(P0./50);
		
		%----------------------- Check convergence ------------------------
		
		I_err = Iac_loop_f - Iac_loop;
		if abs(I_err) < Iac_tol %----- If less than tolerance, converged --
			
			barprint("Converged!");
			
			% Save result
			Iac_exp(idx) = Iac_loop;
			VF_exp(idx) = VF;
			
			% Move to next point
			break;
		else %-------------------- Otherwise, update current guess --------
			
			% Check if last change was positive or negative
			if delta == NON_INIT
				if I_err > 1
					delta = INCREASE;
				else
					delta = DECREASE;
				end
			elseif delta == INCREASE
				if I_err < 1
					conv_fac = conv_fac/2;
					delta = DECREASE;
					displ("--> Update convergence factor");
				end
			else % was DECREASE
				if I_err > 1
					conv_fac = conv_fac/2;
					delta = INCREASE;
					displ("--> Update convergence factor");
				end
			end
			
			displ("Convergence Error: ", I_err*1e6, " uA");
			
			% Change next guess
			Iac_loop = Iac_loop + conv_fac*I_err;
		end
		
	end
	
end

% Save frequencies
freqs_exp = freqs;

%% Plot Measured Data

% Import data
load(dataset_path("DS6_10GHzFreqSweep_FF1.mat"));

SG_pwr = ld(1).SG_power_dBm;

% Create data arrays
all_freq = [ld.SG_freq_Hz];
S21 = zeros(1, numel(ld));

% Get conversion parameters
a2 = sqrt(cvrt(-10, 'dBm', 'W'));
a_SG = sqrt(cvrt(SG_pwr, 'dBm', 'W'));

% Get unique frequencies
freqs = unique(all_freq);
avg_S21 = zeros(1, numel(freqs));
std_S21 = zeros(1, numel(freqs));

% Get mean  of VNA data for b1a2
for idx = 1:numel(S21)
	
	% Get mean of data
	b1a2 = mean(ld(idx).VNA_data.data(1,:));

	% Convert a1b2 to S21
	S21(idx) = abs(b1a2).*a2./a_SG;
	
end

% Average by frequency
idx = 0;
for f = freqs
	idx = idx + 1;
	
	% Get mask
	I = (f == all_freq);
	
	avg_S21(idx) = mean(S21(I));
	std_S21(idx) = std(S21(I));
	
end

% Get V and I
V = avg_S21.*a_SG.*sqrt(50);
Iac = V./50.*sqrt(2);
% Iac = avg_S21./sqrt(50);

warning("This is calculating I at VNA, not along chip!");

figure(1);
hold off;
plot(freqs./1e9, Iac.*1e3, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 0.2, 'Color', [0, .6, 0]);
hold on;
plot(freqs_exp./1e9, Iac_initial_guess.*1e3, 'LineStyle', '-', 'LineWidth', 0.5, 'Color', [0.5, 0, 0]);
plot(freqs_exp./1e9, Iac_exp.*1e3, 'LineStyle', '-', 'LineWidth', 1.3, 'Color', [0, 0, 0.6]);
grid on;
xlabel("Frequency (GHz)");
ylabel("VNA Current (mA)")
title("Measurement vs Expectation Comparison")
force0y;
legend("Measured Current", "Expected Current (Without Nonlin.)", "Expected Current (With Nonlin.)");

figure(2);
hold off;
plot(freqs_exp./1e9, VF_exp, 'LineStyle', ':', 'Marker', '.', 'LineWidth', 1, 'Color', [.6, .7, 0]);
xlabel("Frequency (GHz)");
ylabel("Velocity Factor (1)");
title("Velocity Factor versus Frequency");
grid on;
ylim([0.301, 0.304]);










