function I = solveIbias_single(Vset)
% SOLVEIBIAS_SINGLE See SOLVEIBIAS
%
%

	% Coefficients
	R_0 = 105.26; %[Ohm]
	q = .190; % [A]
	Vtol = 1e-3; % [V]
	Istep = 50e-3;
	step_change = 5; % Divisor to modify step when it changes
	
	num_iter = 0;
	
	% Initial guess (perfect source)
	I_g = Vset/R_0;
	
	% Find error
	V_g = I_g^3/q^2 + 20*I_g;
	Verr = Vset - V_g;
	
	direction = 'x'; % 'd' for decrease, 'i' for increase, 'x' for reset
	while abs(Verr) > Vtol
		
% 		displ("Iguess = ", I_g, ", Error = ", Verr);
		
		% Increment counter
		num_iter = num_iter + 1;
		
		% Update guess
		if Verr > 0
			if direction == 'd'
				Istep = Istep/step_change;
% 				displ("    Changed step size to ", Istep*1e6, " uA");
				direction = 'x';
			else
				direction = 'i';
			end
			I_g = I_g + Istep;
		else
			if direction == 'i'
				Istep = Istep/step_change;
% 				displ("    Changed step size to ", Istep*1e6, " uA");
				direction = 'x';
			else
				direction = 'd';
			end
			I_g = I_g - Istep;
		end
		
		% Find error
		V_g = I_g^3/q^2 + 20*I_g;
		Verr = Vset - V_g;
		
	end
	
% 	displ("Number of guesses: ", num_iter);

	I = I_g;
end