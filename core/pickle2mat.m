function ld = pickle2mat(pickle_fn, mat_fn)

	% Check compatible version
	try
		vstr = version('-release'); % Get release version
		version_year = str2double(vstr(1:end-1)); % Get version year
	catch
		warning("Failed to detect MATLAB version");
	end
	if version_year <= 2020
		warning("This function is not compatible with this version of MATLAB. Known to work with 2022b.");
		return;
	elseif version_year < 2022
		warning("This function may not be compatible with this version of MATLAB. Known to work with 2022b");
	end


	fid = py.open(pickle_fn, 'rb');
	data = py.pickle.load(fid);
	
	% Convert to cell array
	cd = cell(data);

	ld = [];

	% Convert each element to a struct
	idx = 0;
	for dp = cd
		idx = idx + 1;

		% Convert to struct
		cd{idx} = struct(dp{1});

		% Convert elements to native type
		cd{idx}.collection_index = double(cd{idx}.collection_index);
		cd{idx}.timestamp = string(cd{idx}.timestamp);
		cd{idx}.harmonic = double(cd{idx}.harmonic);
		cd{idx}.harmonic = double(cd{idx}.harmonic);

		% Get complex values from arcane format
		ts_cells = cell(cd{idx}.MFLI_voltage);
		re = double(ts_cells{1});
		im = double(ts_cells{2});
		cd{idx}.MFLI_voltage = complex(re, im);
		
		% Handle nested struct
		vna_data = struct(cd{idx}.VNA_data);
		vna_data.label = string(vna_data.label);
		vna_data.freq_Hz = double(vna_data.freq_Hz);
		vna_data.freq_Hz = vna_data.freq_Hz(1:2); % Remove last element - unused

		% Go through tedious process of converting numpy array to an array
		vdc = cell(vna_data.data);
		vdd = double(vdc{1});
		vna_data.data = vdd;

		cd{idx}.VNA_data = vna_data; % Save back to strucutre
		
		% Save as a list of structs instead of cells, because cells are
		% annoying.
		ld = [ld, cd{idx}];

		%Comment to fix MATLAB's new formatting breaker :'(
	end

	save(mat_fn, 'ld');
end