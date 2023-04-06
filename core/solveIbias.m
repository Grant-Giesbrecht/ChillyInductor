function Is = solveIbias(Vset)
% SOLVEIBIAS Solves for the bias current in kinetic inductance
%
%

	% Output variables
	Is = zeros(1, numel(Vset));

	% Loop over all
	idx = 1;
	for v = Vset
		Is(idx) = solveIbias_single(v);
		idx = idx + 1;
	end

end