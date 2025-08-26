function full_path = dataset_path(filename)
% DATASET_PATH Reads the conf file and returns the patht to the requested
% dataset.
%

	% Load configuration file
% 	conf = load_gconf(fullfile("..", "chilly.gconf"));
	conf = load_gconf(fullfile(gethomedir(), "Documents" , "GitHub", "ChillyInductor", "RP-21 Scripts" , "chilly.gconf"));
	
% 	d = dir(conf.data_dir)
	
	% Append filename
	if ispc
		full_path = fullfile(conf.data_dir_pc, filename);
	else
		full_path = fullfile(conf.data_dir, filename);
	end
end