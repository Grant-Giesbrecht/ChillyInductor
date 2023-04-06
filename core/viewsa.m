function viewsa(ld, title_base)
% VIEWSA View a struct array
%
% Created to display datasets from the cryostat.

	if ~exist('title_base', 'var')
		title_base = 'Struct Array';
	end

	NA_str = "---";
	
	mt = MTable();
	mt.row(["Parameter", "No. Unique", "Type", "Min", "Max", "Size"]);
	
	% Get list of field names
	names = ccell2mat(fieldnames(ld));

	% Scan over each field
	for n = names
		
		% Get data as list
		x = [ld.(n)];
		
		% Get values for each column
		if isnumeric(x)
			str_min = num2fstr(min(x));
			str_max = num2fstr(max(x));
			str_unqe = num2str(numel(unique(x)));
		else
			str_min = NA_str;
			str_max = NA_str;
			str_unqe = NA_str;
		end
		[r,c] = size(x);
		str_size = strcat("[", num2str(r), "x", num2str(c), "]");
		str_type = string(class(x));
		
		% Print to row
		mt.row([n, str_unqe, str_type, str_min, str_max, str_size]);
		
	end
	
	% Update title
	mt.title(strcat(title_base, " Summary"));
	
	displ(mt.str());
end