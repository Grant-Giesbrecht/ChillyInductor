function ds = pickle2mat_v2(pickle_fn, mat_fn)
	
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
	
	% Convert to struct
	ds = struct(data);
	
	% Convert configuration data
	ds.configuration = struct(ds.configuration);
	ds.configuration.description = string(ds.configuration.description);
	ds.configuration.RF_power = double(ds.configuration.RF_power);
	ds.configuration.frequency = double(ds.configuration.frequency);
	ds.configuration.harmonics = double(ds.configuration.harmonics);
	ds.configuration.bias_V = double(ds.configuration.bias_V);
	ds.configuration.intersperce_0bias = logical(ds.configuration.intersperce_0bias);
	ds.configuration.duplicate_reverse = logical(ds.configuration.duplicate_reverse);
	ds.configuration.num_pts = double(ds.configuration.num_pts);
	ds.configuration.ds_name = string(ds.configuration.ds_name);

	% Convert source power metric
	ds.VNA_source_pwr_dBm = double(ds.VNA_source_pwr_dBm);
	
	cd = cell(ds.dataset);

	% ld = [];
	clear ld;

	% Convert each element to a struct
	idx = 0;
	wb = waitbar(0, 'Converting file (0%).');
	t0 = tic();
	for dp = cd
		
		if toc(t0) > 0.5
			t0 = tic();
			waitbar(idx/numel(cd), wb, strcat("Converting file (", num2str(round(idx/numel(cd)*1000)/10), "%)."))
		end

		idx = idx + 1;

		% Convert to struct
		cd{idx} = struct(dp{1});

		% Convert elements to native type
		cd{idx}.collection_index = double(cd{idx}.collection_index);
		cd{idx}.timestamp = string(cd{idx}.timestamp);
		cd{idx}.harmonic = double(cd{idx}.harmonic);
		cd{idx}.offset_V = double(cd{idx}.offset_V);
		cd{idx}.MFLI_voltage_V = double(cd{idx}.MFLI_voltage_V);
		cd{idx}.MFLI_voltage_deg = double(cd{idx}.MFLI_voltage_deg);
		cd{idx}.MFLI_voltage = polcomplex(cd{idx}.MFLI_voltage_V, cd{idx}.MFLI_voltage_deg, 'Unit', 'degrees');
		cd{idx}.SG_power_dBm = double(cd{idx}.SG_power_dBm);
		cd{idx}.SG_freq_Hz = double(cd{idx}.SG_freq_Hz);
		cd{idx}.temp_K = double(cd{idx}.temp_K);

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
		if idx == 1
			ld(numel(cd)) = cd{idx};
			ld(1)= cd{idx};
		else
			ld(idx)= cd{idx};
		end
		% ld = [ld, cd{idx}];

		%Comment to fix MATLAB's new formatting breaker :'(
	end
	
	close(wb);

	ds.dataset = ld;
	
	displ("  Saving MAT file...");
	save(mat_fn, 'ds');
end