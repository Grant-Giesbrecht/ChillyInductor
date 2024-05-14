% Describe files to loop over
prefix = 'meas_through_';
S11_postfix = '_S11.s2p';
S21_postfix = '_S21.s2p';
files = ["2K8", "3K0", "4K0", "5K0", "RTP"];
temps = [2.8, 3, 4, 5, 293];

DATA_PATH = fullfile('/','Users','grantgiesbrecht','MEGA','NIST Datasets','group3_2023pub','temp_sweep_through');

T_s11 = [];
T_s21 = [];

figure(1);
hold off;

figure(2);
hold off;

f1l = {};
f2l = {};

% Loop over all files
for fidx = 1:numel(files)
	
	% Read S2P files - S11
	skip_s11 = false;
	try
		S11_rf = sparameters(fullfile(DATA_PATH, prefix+files(fidx) + S11_postfix));
	catch
		skip_s11 = true;
	end
	if ~skip_s11
		T_s11 = [T_s11, temps(fidx)];
		f1l{end+1} = "T = " + num2str(temps(fidx)) + " K";
		
		% Plot data
		figure(1);
		plot(flatten(S11_rf.Frequencies./1e9), S11_dB);
		hold on;
	end
	
	% Read S2P files - S21
	skip_s21 = false;
	try
		S21_rf = sparameters(fullfile(DATA_PATH, prefix+files(fidx) + S21_postfix));
	catch
		skip_s21 = true;
	end
	if ~skip_s21
		T_s21 = [T_s21, temps(fidx)];
		f2l{end+1} = "T = " + num2str(temps(fidx)) + " K";
		
		figure(2);
		plot(flatten(S21_rf.Frequencies./1e9), S21_dB);
		hold on;
	end
	
	S11_dB = lin2dB(abs(flatten(S11_rf.Parameters(1, 1, :))));
	S21_dB = lin2dB(abs(flatten(S11_rf.Parameters(2, 1, :))));
	
	
	
	

end

figure(1);
grid on;
xlabel("Frequency (GHz)");
title("Reflection Coefficient vs Temperature");
ylabel("S_{11} (dB)");
legend(f1l{:});


figure(2);
grid on;
xlabel("Frequency (GHz)");
title("Transmission Coefficient vs Temperature");
ylabel("S_{21} (dB)");
legend(f2l{:});





















