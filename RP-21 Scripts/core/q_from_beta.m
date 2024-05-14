function [mean_q_vs_freq, std_q_vs_freq, all_qs] = q_from_beta(ds, P_RF, analysis_freqs, conditions)
	
	%------------------- Transfer conditions to local vars ----------------
	
	LIMIT_BIAS = conditions.LIMIT_BIAS;
	
	bias_max_A = conditions.bias_max_A;
	bias_min_A = conditions.bias_min_A;
	
	len = conditions.len; % meters
	Vp0 = conditions.Vp0; % m/s
	iv_conv = conditions.iv_conv; % A/V
	
	%------------------- Run q analysis -----------------------------------
	
	% Prepare data arrays
	all_qs = [];
	mean_q_vs_freq = zeros(1, numel(analysis_freqs));
	std_q_vs_freq = zeros(1, numel(analysis_freqs));
% 	I_vals = zeros(1, numel(analysis_freqs));
% 	phase_vals = zeros(1, numel(analysis_freqs));
	
	% Process each frequency - get q
	idx = 0;
	for f = analysis_freqs
		idx = idx + 1;
		
		% Apply drift correction, unwrap phase, and normalize
		[cp, vb] = getCorrectedPhase_V2(ds, P_RF, f, false);
		
		% Get bias currents
		Ibias = iv_conv.*vb;
		
		% Limit bias
		if LIMIT_BIAS
			Ipass = (abs(Ibias) >= bias_min_A) & (abs(Ibias) <= bias_max_A);
			cp = cp(Ipass);
			vb = vb(Ipass);
			Ibias = Ibias(Ipass);
		end
		
		% Remove zero point (don't divide by zero)
		I_calc = Ibias(cp ~= 0);
		cp_calc = cp(cp ~= 0);
		
		% Calculate q
		q = sqrt(180.*f.*len./abs(cp_calc)./Vp0).*abs(I_calc);
		
		% Save to structs
		mean_q_vs_freq(idx) = mean(q);
		std_q_vs_freq(idx) = std(q);
		
% 		I_vals(idx) = Ibias;
% 		phase_vals(idx) = cp;
		
		all_qs = [all_qs, q];
	end
	
end