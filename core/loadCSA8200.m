function [t, R] = loadCSA8200(fileIn, dt, t0, matFile)
    
    % Get default arguments
    if ~exist('dt', 'var') || dt <= 0
       dt = -1; 
    end
    if ~exist('t0', 'var')
       t0 = 0;
    end
    if ~exist('matFile', 'var')
       matFile = "";
    end
    
    % Save original t0 spec
    t0_0 = t0;
    
    % Open file
    fid = fopen(fileIn);
    if fid == -1
       warning("Failed to open file.");
    end
    
    % Loop over entire file contents
    t = [];
    R = [];
    while (~feof(fid))
        
        % Read line
        sline = fgetl(fid);
        
        % Parse words
        words = parseIdx(sline, [",", " "]);
        
        % Generate blank array
        R_ = zeros(1, numel(words));
        
        % Interpret each word
        for widx = 1:numel(words)
            
            % Convert to double
            try
                R_(widx) = str2double(words(widx).str);
            catch
                R_(widx) = NaN;
            end
        end
        
        % Generate time array
        if dt == -1
            tx = numel(words)+t0;
            t_ = t0:1:(tx-1);
            t0 = tx; % Update starting point for next line
        else
            tx = dt*numel(words)+t0;
            t_ = t0:dt:(tx-dt);
            t0 = tx; % Update starting point for next line
        end
        
        % Append to master arrays
        t = cat(2, t, t_);
        R = cat(2, R, R_);
        
    end
    
    % Save MAT file if requested
    if matFile ~= ""
        source_file = fileIn;
        conv_date = date();
        t0 = t0_0;
        save(matFile, 't', 'R', 'source_file', 'conv_date', 't0', 'dt');
    end
    
end