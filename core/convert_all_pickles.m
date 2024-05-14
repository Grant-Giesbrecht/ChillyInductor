function ds = convert_all_pickles(pickle_fn, mat_fn)
	
	%---- Get filenames to convert - end in pkl and don't have mat equivs.
	% Thanks Copilot <3
	
	% Get the current working directory
	workingDir = pwd;
	
	% List all files with the extension '.pkl'
	pklFiles = dir(fullfile(workingDir, '*.pkl'));
	
	% Initialize an empty cell array to store valid filenames
	validFilenames = {};
	
	% Iterate through each '.pkl' file
	for i = 1:length(pklFiles)
    	pklFilename = pklFiles(i).name;
    	
    	% Check if a corresponding '.mat' file exists
    	matFilename = strrep(pklFilename, '.pkl', '.mat');
    	if ~exist(fullfile(workingDir, matFilename), 'file')
        	validFilenames{end+1} = pklFilename;
    	end
	end
	
	% Convert the valid filenames
	fprintf('Files identified for conversion:\n');
	for i = 1:length(validFilenames)
		fprintf('\tci%s\n', validFilenames{i});
	end

	fprintf('\nConverting all lone PKL files:\n');
	for i = 1:length(validFilenames)
    	pickle2mat_v2(validFilenames{i});
	end
	
end