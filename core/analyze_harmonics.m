function [h, out_data] = analyze_harmonics(ds, fig_type, settings, varargin)
%ANALYZE_HARMONICS Analyze V2 harmonic data and generate figures.
%
%	ANALYZE_HARMONICS(DS, FIG_TYPE, SETTINGS) Generate the specified type of
%	figure with the specified settings.
%	
	
%   See also ASIN, SIND, SINPI.
%
%
%
%

	%% Parse optional arguments
	
	p = inputParser;
    p.KeepUnmatched = true;
	p.addParameter('Fig', 1, @isnumeric );
	p.addParameter('Hold', false, @islogical );
	p.addParameter('LegendPrefix', "", @(x) isstring(x) || ischar(x) );
	p.addParameter('Color1', [0, 0, 0.7], @isnumeric );
	p.addParameter('Color2', [0, 0.6, 0], @isnumeric );
	p.addParameter('Color3', [0.6, 0, 0], @isnumeric );
	p.addParameter('EqualizeScales', false, @islogical );
	p.addParameter('StatusUpdates', false, @islogical );
	p.addParameter('UseSystemCE', false, @islogical ); % I'm defining System CE as conversion efficiency of the entire system, ie. P2/Pset, not P2/SUM(Pn). Other would be chip CE where reflection is removed.
	
    p.parse(varargin{:});
	p = p.Results;
	
	% Add comma to prefix
	if ~strcmp(p.LegendPrefix, "")
		p.LegendPrefix = p.LegendPrefix + ", ";
	end
	
	out_data = [];
	
	%% Copy settings to local variables
		
	P_RF = settings.P_RF; % dBm
	FREQ = settings.FREQ; % Hz
		
	NORMAL_VOLTAGE = settings.NORMAL_VOLTAGE;
	% NORMAL_VOLTAGE = 0.002;
	
	CMAP = settings.CMAP;
	
	powers_dBm = ds.configuration.RF_power;
	freqs = ds.configuration.frequency;
	
	% Prepare conditions struct
	c = defaultConditions();
	c.SG_power = P_RF;
	c.convert_to_W = 1;
	c.f0 = FREQ;
	c.Vnorm = NORMAL_VOLTAGE;
	
	if strcmpi(fig_type, 'harm_power')
	%% Extract data for Selected Single Condition
		
		% function [harm_struct, normal, Vsweep] = getHarmonicSweep_v2(rich_data, c, keep_normal)
		[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
		
		% Prepare figure
		figure(p.Fig);
		subplot(1, 1, 1);
		if ~p.Hold
			hold off;
		end
		
		% Plot data
		h = plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', p.Color1, 'DisplayName', p.LegendPrefix+'Fundamental');
		hold on;
		plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', p.Color2, 'DisplayName', p.LegendPrefix+'2nd Harmonic');
		plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', p.Color3, 'DisplayName', p.LegendPrefix+'3rd Harmonic');
		grid on;
		xlabel("Bias Voltage (V)");
		ylabel("Harmonic Power at Chip Output (dBm)");
		title("Freq = " + num2str(FREQ./1e9) + " GHz, P = " + num2str(P_RF) + " dBm");
		legend();
	
	
	elseif strcmpi(fig_type, 'harm_power_vs_pin')
	%% Extract data for Power at Each Harmonic, at each Pin
	
		% Prepare figure
		figure(p.Fig);		
		if ~p.Hold
			subplot(1, 3, 1);
			hold off;
			subplot(1, 3, 2);
			hold off;
			subplot(1, 3, 3);
			hold off;
		else
			subplot(1, 3, 1);
			hold on;
			subplot(1, 3, 2);
			hold on;
			subplot(1, 3, 3);
			hold on;
		end
		
		% Generate graph
		LL4 = {};
		CM = resamplecmap(CMAP, numel(powers_dBm));
		idx = 0;
		for pwr = powers_dBm
			idx = idx + 1;
			
			% Update conditions
			c.SG_power = pwr;
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
			
			figure(p.Fig);
			subplot(1, 3, 1);
			h = plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
			hold on;
			subplot(1, 3, 2);
			plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
			hold on;
			subplot(1, 3, 3);
			plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :));
			hold on;
			
			LL4 = [LL4(:)', {"P = "+num2str(pwr) + " dBm"}];
			
		end
		
		figure(p.Fig);
		subplot(1, 3, 1);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("Fundamental's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
		grid on;
		subplot(1, 3, 2);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("2nd Harmonic's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
		grid on;
		subplot(1, 3, 3);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("3rd Harmonic's RF Power Dependence, f="+num2str(FREQ/1e9) + " GHz");
		grid on;
		
	elseif strcmpi(fig_type, 'harm_power_vs_freq')
	%% Extract data for Power at Each Harmonic, at each fund. freq
	
		figure(p.Fig);
		if ~p.Hold
			subplot(1, 3, 1);
			hold off;
			subplot(1, 3, 2);
			hold off;
			subplot(1, 3, 3);
			hold off;
		else
			subplot(1, 3, 1);
			hold on;
			subplot(1, 3, 2);
			hold on;
			subplot(1, 3, 3);
			hold on;
		end
		
		LL4 = {};
		CM = resamplecmap(CMAP, numel(freqs));
		idx = 0;
		for f = freqs
			idx = idx + 1;
			
			% Update conditions
			c.f0 = f;
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
			
			figure(p.Fig);
			subplot(1, 3, 1);
			h(idx) = plot(Vsweep, cvrt(abs(harm_struct.h1), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :), 'DisplayName', num2str(f/1e9));
			hold on;
			subplot(1, 3, 2);
			h(idx+numel(freqs)) = plot(Vsweep, cvrt(abs(harm_struct.h2), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :), 'DisplayName', num2str(f/1e9));
			hold on;
			subplot(1, 3, 3);
			h(idx+numel(freqs)*2) = plot(Vsweep, cvrt(abs(harm_struct.h3), 'W', 'dBm'), 'LineStyle', '--', 'Marker', 'o', 'Color', CM(idx, :), 'DisplayName', num2str(f/1e9));
			hold on;
			
			% Apply custom data tips
			h(idx).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(idx).DataTipTemplate.DataTipRows(2).Label = "P_{Out} (dBm)";
			h(idx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('f_0 (GHz)',repmat({h(idx).DisplayName},size(h(idx).XData)));
			
			% Apply custom data tips
			h(idx+numel(freqs)).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(idx+numel(freqs)).DataTipTemplate.DataTipRows(2).Label = "P_{Out} (dBm)";
			h(idx+numel(freqs)).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('f_0 (dBm)',repmat({h(idx).DisplayName},size(h(idx).XData)));
			
			% Apply custom data tips
			h(idx+numel(freqs)*2).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(idx+numel(freqs)*2).DataTipTemplate.DataTipRows(2).Label = "P_{Out} (dBm)";
			h(idx+numel(freqs)*2).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('f_0 (dBm)',repmat({h(idx).DisplayName},size(h(idx).XData)));
			
			LL4 = [LL4(:)', {"f0 = "+num2str(f/1e9) + " GHz"}];
			
		end
		
		figure(p.Fig);
		subplot(1, 3, 1);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("Fundamental's Frequency Dependence, P="+num2str(P_RF) + " dBm");
		grid on;
		subplot(1, 3, 2);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("2nd Harmonic's Frequency Dependence, P="+num2str(P_RF) + " dBm");
		grid on;
		subplot(1, 3, 3);
		legend(LL4{:});
		xlabel("Bias Voltage (V)");
		ylabel("Power at Chip Ouptut (dBm)");
		title("3rd Harmonic's Frequency Dependence, P="+num2str(P_RF) + " dBm");
		grid on;
		
	elseif strcmpi(fig_type, 'max_ce_vs_freq_power')
	%% Extract maximum CE over freq, with a trace for each power
		
		% Prepare figure
		figure(p.Fig);
		if ~p.Hold
			subplot(2, 1, 1);
			hold off;
			subplot(2, 1, 2);
			hold off;
		else
			subplot(2, 1, 1);
			hold on;
			subplot(2, 1, 2);
			hold on;
		end
		
		% Scan over all powers
		CM = resamplecmap(CMAP, numel(powers_dBm));
		LL4 = cell(1, numel(powers_dBm));
		for pidx = 1:numel(powers_dBm)
			pwr = powers_dBm(pidx);
			
			% Generate local conditions struct
			c = defaultConditions();
			c.SG_power = pwr;
			c.convert_to_W = 1;
			c.Vnorm = NORMAL_VOLTAGE;
			
		% 	multiWaitbar('Generate Figure 4', (pidx-1)/numel(powers_dBm));
			
			CE2 = zeros(1, numel(freqs));
			CE3 = zeros(1, numel(freqs));
			CE2_sys = zeros(1, numel(freqs));
			CE3_sys = zeros(1, numel(freqs));
			
			% Scan over all frequencies
			idx = 0;
			for f = freqs
				idx = idx + 1;
								
				% Update conditions
				c.f0 = f;
				
				% Extract data
				[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
				
				% Calculate conversion efficiency
				CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
				CE3_ = abs(harm_struct.h3)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
				CE2sys_ = abs(harm_struct.h2)./cvrt(pwr, 'dBm', 'W').*100;
				CE3sys_ = abs(harm_struct.h3)./cvrt(pwr, 'dBm', 'W').*100;
				[CE2(idx), mi2] = max(CE2_);
				[CE3(idx), mi3] = max(CE3_);
				[CE2_sys(idx), mi2s] = max(CE2sys_);
				[CE3_sys(idx), mi3s] = max(CE3sys_);
			end
			
			% Save output data
			out_data.("P"+num2str(pwr)+"dBm").CE2_chip = CE2;
			out_data.("P"+num2str(pwr)+"dBm").CE3_chip = CE3;
			out_data.("P"+num2str(pwr)+"dBm").CE2_sys = CE2_sys;
			out_data.("P"+num2str(pwr)+"dBm").CE3_sys = CE3_sys;
			
			% Select appropriate CE
			if p.UseSystemCE
				CE2_sel = CE2_sys.*8;
				CE3_sel = CE3_sys.*8;
				ce_type_str = "System";
			else
				CE2_sel = CE2;
				CE3_sel = CE3;
				ce_type_str = "Chip";
			end
			
			% Plot this power level
			figure(p.Fig);
			subplot(2, 1, 1);
			h(pidx) = plot(freqs./1e9, CE2_sel, 'LineStyle', '--', 'Marker', '+', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
			hold on;
			subplot(2, 1, 2);
			h(pidx+numel(powers_dBm)) = plot(freqs./1e9, CE3_sel, 'LineStyle', '--', 'Marker', '+', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
			hold on;
			
			% Apply custom data tips
			h(pidx).DataTipTemplate.DataTipRows(1).Label = "Frequency (GHz)";
			h(pidx).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
			h(pidx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({h(pidx).DisplayName},size(h(pidx).XData)));
			
			% Apply custom data tips
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(1).Label = "Frequency (GHz)";
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({h(pidx).DisplayName},size(h(pidx).XData)));
			
			LL4{pidx} ="P = "+num2str(pwr) + " dBm";
			
			% Print Update
			if p.StatusUpdates
				displ('  --> (Fig. ', p.Fig, ') Finished power level ',pidx, ' of ', numel(powers_dBm), ' (', 100*(pidx)/numel(powers_dBm), '%) [mode:max_ce_vs_freq_power]')
			end
			
		end
		% multiWaitbar('CloseAll');

		figure(p.Fig);
		subplot(2, 1, 1);
		xlabel("Frequency (GHz)");
		ylabel("Maximum Conversion Efficiency (%)");
		title("2nd Harmonic "+ce_type_str+" Conversion Efficiency");
		grid on;
		legend(LL4{:});
		subplot(2, 1, 2);
		xlabel("Frequency (GHz)");
		ylabel("Maximum Conversion Efficiency (%)");
		title("3rd Harmonic "+ce_type_str+" Conversion Efficiency");
		grid on;
		legend(LL4{:});
		
	elseif strcmpi(fig_type, 'ce_vs_bias_power')
		
		% Prepare graph
		figure(p.Fig);
		if ~p.Hold
			subplot(1, 2, 1);
			hold off;
			subplot(1, 2, 2);
			hold off;
		else
			subplot(1, 2, 1);
			hold on;
			subplot(1, 2, 2);
			hold on;
		end
		
		% Scan over all powers
		CM = resamplecmap(CMAP, numel(powers_dBm));
		LL4 = cell(1, numel(powers_dBm));
		for pidx = 1:numel(powers_dBm)
			pwr = powers_dBm(pidx);
			
			% Generate local conditions struct
			c = defaultConditions();
			c.SG_power = pwr;
			c.convert_to_W = 1;
			c.Vnorm = NORMAL_VOLTAGE;
			c.f0 = FREQ;
			
		% 	multiWaitbar('Generate Figure 4', (pidx-1)/numel(powers_dBm));
			
			CE2 = zeros(1, numel(freqs));
			CE3 = zeros(1, numel(freqs));
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
			
			% Calculate conversion efficiency
			CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			CE3_ = abs(harm_struct.h3)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			
			figure(p.Fig);
			subplot(1, 2, 1);
			h(pidx) = plot(Vsweep, CE2_, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
			hold on;
			subplot(1, 2, 2);
			h(pidx+numel(powers_dBm)) = plot(Vsweep, CE3_, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
			hold on;
			
			LL4{pidx} ="P = "+num2str(pwr) + " dBm";
			
			% Apply custom data tips
			h(pidx).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(pidx).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
			h(pidx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({h(pidx).DisplayName},size(h(pidx).XData)));
			
			% Apply custom data tips
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
			h(pidx+numel(powers_dBm)).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({h(pidx).DisplayName},size(h(pidx).XData)));
			
		end
		% multiWaitbar('CloseAll');
		
		figure(p.Fig);
		subplot(1, 2, 1);
		xlabel("Bias Voltage (V)");
		ylabel("Conversion Efficiency (%)");
		title("2nd Harmonic, f = " + num2str(FREQ/1e9) + " GHz");
		grid on;
		legend(LL4{:});
		subplot(1, 2, 2);
		xlabel("Bias Voltage (V)");
		ylabel("Conversion Efficiency (%)");
		title("3rd Harmonic, f = " + num2str(FREQ/1e9) + " GHz");
		grid on;
		legend(LL4{:});
	elseif strcmpi(fig_type, 'vmfli_vs_bias')
		
		% Prepare graph
		figure(p.Fig);
		if ~p.Hold
			subplot(1, 1, 1);
			hold off;
		else
			subplot(1, 1, 1);
			hold on;
		end
		
		% Scan over all powers
		CM = resamplecmap(CMAP, numel(powers_dBm));
		LL4 = cell(1, numel(powers_dBm));
		for pidx = 1:numel(powers_dBm)
			pwr = powers_dBm(pidx);
			
			% Generate local conditions struct
			c = defaultConditions();
			c.SG_power = pwr;
			c.convert_to_W = 1;
			c.Vnorm = NORMAL_VOLTAGE;
			c.f0 = FREQ;
			
		% 	multiWaitbar('Generate Figure 4', (pidx-1)/numel(powers_dBm));
			
			CE2 = zeros(1, numel(freqs));
			CE3 = zeros(1, numel(freqs));
			
			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
			
			figure(p.Fig);
			h(pidx) = plot(Vsweep, norm.V.*1e3, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
			hold on;
			
			LL4{pidx} ="P = "+num2str(pwr) + " dBm";
			
			% Apply custom data tips
			h(pidx).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
			h(pidx).DataTipTemplate.DataTipRows(2).Label = "V_{MFLI} (V)";
			h(pidx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({h(pidx).DisplayName},size(h(pidx).XData)));
		end
		% multiWaitbar('CloseAll');
		
		figure(p.Fig);
		xlabel("Bias Voltage (V)");
		ylabel("MFLI Voltage (mV)");
		title("DC Voltage Across Chip, f = " + num2str(FREQ/1e9) + " GHz");
		grid on;
		legend(LL4{:});
	elseif strcmpi(fig_type, 'ce2_vs_bias_power_freq')
		
		% Get number of rows and columns
		num_plots = numel(FREQ);
		cols = ceil(sqrt(num_plots));
		rows = ceil(num_plots/cols);
		
		% Prepare graph
		figure(p.Fig);
		if ~p.Hold
			for np = 1:num_plots
				subplot(rows, cols, np);
				hold off;
			end
		else
			for np = 1:num_plots
				subplot(rows, cols, np);
				hold on;
			end
		end
		
		% Scan over each subplot
		for np = 1:num_plots
			
			% Scan over all powers
			CM = resamplecmap(CMAP, numel(powers_dBm));
			LL4 = cell(1, numel(powers_dBm));
			for pidx = 1:numel(powers_dBm)
				pwr = powers_dBm(pidx);
				
				% Generate local conditions struct
				c = defaultConditions();
				c.SG_power = pwr;
				c.convert_to_W = 1;
				c.Vnorm = NORMAL_VOLTAGE;
				c.f0 = FREQ(np);
				
				% Extract data
				[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);
				
				% Calculate conversion efficiency
				CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
				
				figure(p.Fig);
				subplot(rows, cols, np);
				hs(pidx) = plot(Vsweep, CE2_, 'LineStyle', '--', 'Marker', 'o', 'Color', CM(pidx, :), 'DisplayName', num2str(pwr));
				hold on;
				
				LL4{pidx} ="P = "+num2str(pwr) + " dBm";
				
				% Apply custom data tips
				hs(pidx).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
				hs(pidx).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
				hs(pidx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({hs(pidx).DisplayName},size(hs(pidx).XData)));
				
			end
			
			% Add to list of handle
			h{np} = hs;
			
			% Label figure
			figure(p.Fig);
			subplot(rows, cols, np);
			xlabel("Bias Voltage (V)");
			ylabel("Conversion Efficiency (%)");
			title("2nd Harmonic, f = " + num2str(FREQ(np)/1e9) + " GHz");
			grid on;
			legend(LL4{:});
			
		end
		
		% Equalize scales
		if p.EqualizeScales

			yl_min = NaN;
			yl_max = NaN;

			% Get extremes
			for handle_cell = h

				% Access graph
				graph = handle_cell{1}(1).Parent;

				yl = graph.YLim();

				% Update extremes
				if isnan(yl_max) || yl(2) > yl_max
					yl_max = yl(2);
				end
				if isnan(yl_min) || yl(1) < yl_min
					yl_min = yl(1);
				end					
			end

			% Apply same limits for all graphs
			for handle_cell = h

				% Access graph
				graph = handle_cell{1}(1).Parent;

				graph.YLim = [yl_min, yl_max];	
			end
		end
		
	elseif strcmpi(fig_type, 'CE_surf_freq_vs_bias')
		
		% Get number of rows and columns
		num_plots = numel(FREQ);
		cols = ceil(sqrt(num_plots));
		rows = ceil(num_plots/cols);
		
		% Prepare graph
		figure(p.Fig);
		if ~p.Hold
			hold off;
		else
			hold on;
		end
		
		% Allocate variables
		[B, F] = meshgrid(ds.configuration.bias_V, freqs./1e9);
		
		% Create power variable
		CE_2D = zeros(size(F));
		
		% Scan over all powers
		for fidx = 1:numel(freqs)

			% Generate local conditions struct
			c = defaultConditions();
			c.SG_power = P_RF;
			c.convert_to_W = 1;
			c.Vnorm = NORMAL_VOLTAGE;
			c.f0 = freqs(fidx);

			% Extract data
			[harm_struct, norm, Vsweep] = getHarmonicSweep_v2(ds, c, false);

			% Calculate conversion efficiency
			CE2_ = abs(harm_struct.h2)./(abs(harm_struct.h1) + abs(harm_struct.h2) + abs(harm_struct.h3)).*100;
			CE_2D(fidx, :) = CE2_;
			
% 			% Apply custom data tips
% 			hs(pidx).DataTipTemplate.DataTipRows(1).Label = "Bias (V)";
% 			hs(pidx).DataTipTemplate.DataTipRows(2).Label = "\eta (%)";
% 			hs(pidx).DataTipTemplate.DataTipRows(end+1) = dataTipTextRow('P_{RF} (dBm)',repmat({hs(pidx).DisplayName},size(hs(pidx).XData)));
			
		end

		% Label figure
		figure(p.Fig);
		h = surf(B, F, CE_2D);
		ylabel("Frequency (GHz)");
		xlabel("Bias Voltage (V)");
		zlabel("Conversion Efficiency (%)");
		title("Conversion Efficiency, P_{RF} = " + num2str(P_RF) + " dBm");
		grid on;
	
	end % END: graph-type if-statement
	
end