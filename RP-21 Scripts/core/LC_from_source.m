function [Ls, Cs] = LC_from_source(DATA_PATH, FILE_POSTFIXES, EXTENSION, S11_PREFIX, S21_PREFIX, conditions)
	
	%--------------------- TRANSFER CONDITIONS TO LOCAL VARS --------------
	
	NULL_NUM = conditions.NULL_NUM; % Which null is being targeted
	NULL_FREQ_LOW_HZ = conditions.NULL_FREQ_LOW_HZ; % Frequency bounds to target
	NULL_FREQ_HIGH_HZ = conditions.NULL_FREQ_HIGH_HZ;
	
	% Lower and upper limits of bands in which to find detuned gamma magnitude
	DETUNE_FREQ_MIN_HZ = conditions.DETUNE_FREQ_MIN_HZ;
	DETUNE_FREQ_MAX_HZ = conditions.DETUNE_FREQ_MAX_HZ;
	
	%--------------------------- SYSTEM DATA --------------------------------
	
	Z0 = 50;
	l_phys = 0.5;
	
	%---------------------------- MODEL OPTIONS ----------------------------
	
	USE_LOSS_MODEL = false;	

	% Iterate over files
	nf = numel(FILE_POSTFIXES);
	Ls = zeros(1, nf);
	Cs = zeros(1, nf);
	Zcs = zeros(1, nf);
	for idx = 1:nf

		% Read files
		S11_raw = sparameters(fullfile(DATA_PATH, S11_PREFIX+FILE_POSTFIXES(idx)+EXTENSION));
		S21_raw = sparameters(fullfile(DATA_PATH, S21_PREFIX+FILE_POSTFIXES(idx)+EXTENSION));
		S11_dB = lin2dB(abs(flatten(S11_raw.Parameters(1, 1, :))));
		S21_dB = lin2dB(abs(flatten(S11_raw.Parameters(2, 1, :))));

		% Get frequency of selected null
		mask = (S11_raw.Frequencies >= NULL_FREQ_LOW_HZ) & (S11_raw.Frequencies <= NULL_FREQ_HIGH_HZ);
		[val, fm_idx] = min(S11_dB(mask));
		masked_freqs = S11_raw.Frequencies(mask);
		null_freq = masked_freqs(fm_idx);

		% Get reflection magnitude in each bin
		gammas = zeros(1, numel(DETUNE_FREQ_MAX_HZ));
		for midx = 1:numel(DETUNE_FREQ_MAX_HZ)

			% Apply mask for range
			mask = (S11_raw.Frequencies >= DETUNE_FREQ_MIN_HZ(midx)) & (S11_raw.Frequencies <= DETUNE_FREQ_MAX_HZ(midx));

			% Find detune/max
			[detune_S11, fm_idx] = max(S11_dB(mask));

			% Calc. masked parameters
			masked_freqs = S11_raw.Frequencies(mask);
			masked_S21 = S21_dB(mask);

			% Calculate S11 and S21 at max/detuned value
			detune_freq = masked_freqs(fm_idx);
			S11m = dB2lin(detune_S11);
			S21m = dB2lin(masked_S21(fm_idx));

			% Estimate reflection coefficient
			if USE_LOSS_MODEL

				delta = S11m^2 + S21m^2;
				L = delta.^0.25;
				R = S11m./L^2;

				displ("R: ", lin2dB(R), " detune_S11: ", detune_S11)
				gammas(midx) = lin2dB(R);

			else
				gammas(midx) = detune_S11;
			end


		end
		G = dB2lin(mean(gammas));

		% Calculate L and C
		L = NULL_NUM.*Z0./(2 .*null_freq.*l_phys).*sqrt((1 + G)./(1 - G));
		C = (NULL_NUM./(2.*l_phys.*null_freq.*sqrt(L))).^2;
		Zchip = Z0.*sqrt((1+G)./(1-G));

		% Save data
		Ls(idx) = L;
		Cs(idx) = C;
		Zcs(idx) = Zchip;

	end	
	
end