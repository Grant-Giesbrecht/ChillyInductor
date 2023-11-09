% files_base = "MainC3";
% files_post = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"];
% files_ext = ".txt";
% Vdc = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.25];
% output_data_file = "TDR_Dataset_1.mat";
% dt = 50e-15;

files_base = "MainC3B";
files_post = ["1", "2", "3", "4", "5", "6", "7", "8"];
files_ext = ".txt";
Vdc = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.25];
output_data_file = "TDR_Dataset_2.mat";
dt = 12.5e-9;


clear tdr_data;

% Scan over all files
for fidx = 1:numel(files_post)
    
    this_file = strcat("C:\Users\sparky\Desktop\LK_Data\TDR\", files_base, files_post(fidx), files_ext);
    
    disp("Processing file " + this_file);
    
    dp = struct();
    
    % Read file
    [t, R] = loadCSA8200(this_file, dt);
    dp.t = t;
    dp.R = R;
    dp.source = this_file;
    dp.resolution_dt = dt;
    dp.Vdc = Vdc(fidx);
    
    % Initialize array
    if fidx == 1
        tdr_data(numel(files_post)) = dp;
    end
    
    % Save data in array
    tdr_data(fidx) = dp;
    
end

% Create mat file
save(output_data_file, 'tdr_data');