function f = uiconversioneffic(rich_data)
% UICONVERSIONEFFIC
%
% (solns, freqs, spr, spc, optBW_param, e_r, d)
%
% Interactive window for conversion efficiency of the device under various
% conditions.
%

	spr= 3;
	spc = 4;
	num_pages = 3;
	ncirc = 20;
	optBW_param = -20;
	nplots = 12;
	
	%============================= Create GUI =============================
	
	%  Create and then hide the UI as it is being constructed.
	if exist('figno', 'var')
		f = uifigure(figno);
		f.Visible = false;
		f.Position = [360,500,450,285];
	else
		f = uifigure('Visible','off','Position',[360,500,450,285]);
	end
	
	% Master grid layout
	master_layout = uigridlayout(f);
	
	% Create panel for controls
	sideCP = uipanel(master_layout, "Title", "UI Options");
	sideCP.Layout.Row = [1, spr]; % Span all rows
	sideCP.Layout.Column = spc+1;
	
	% Create tab bar
% 	tabbar_p = uipanel(master_layout);
% 	tabbar_p.Layout.Row = 1; % Span all rows
% 	tabbar_p.Layout.Column = [1, spc];
	
	tabbar = uibuttongroup(master_layout);
	tabbar.Layout.Row = 1;
	tabbar.Layout.Column = [1, spc];
	
	% Create tab buttons - 1
	tab1_ctrl = uitogglebutton(tabbar);
	tab1_ctrl.Position = [10, 10, 100, 22];
	tab1_ctrl.Text = "Conversion Effic.";
	
	tab2_ctrl = uitogglebutton(tabbar);
	tab2_ctrl.Position = [160, 10, 100, 22];
	tab2_ctrl.Text = "Calc. q Conf.";
	
	tab3_ctrl = uitogglebutton(tabbar);
	tab3_ctrl.Position = [310, 10, 100, 22];
	tab3_ctrl.Text = "Distrib. Reactance";
	
	
	% Control grid layout
	sideCP_glayout = uigridlayout(sideCP);
	sideCP_glayout.RowHeight = {'fit', 'fit', 'fit', '1x'};
	
	% Create Page Selector Text
	hPageText = uilabel(sideCP_glayout);
	hPageText.Layout.Row = 1;
	hPageText.Layout.Column = 1;
	hPageText.Text = "View Page:";
	hPageText.FontSize = 12;
	hPageText.HorizontalAlignment = 'right';
	
	% Create page selector dropdown
	hPageMenu  = uidropdown(sideCP_glayout);
	hPageMenu.Items = cellstr(string(1:1:num_pages));
	hPageMenu.Layout.Row = 1;
	hPageMenu.Layout.Column = 2;
	hPageMenu.ValueChangedFcn = @(dd, event) page_menu_callback(dd, event);
	
	% Create Circuit Selector Text
	hCircText = uilabel(sideCP_glayout);
	hCircText.Layout.Row = 2;
	hCircText.Layout.Column = 1;
	hCircText.Text = "Print Circuit:";
	hCircText.FontSize = 12;
	hCircText.HorizontalAlignment = 'right';
	
	% Create circuit selector dropdown
	hCircMenu  = uidropdown(sideCP_glayout);
	hCircMenu.Items = cellstr(string(1:1:ncirc));
	hCircMenu.Layout.Row = 2;
	hCircMenu.Layout.Column = 2;
	
	% Create circuit print button
	hCircButton = uibutton(sideCP_glayout);
	hCircButton.Text = "Print";
	hCircButton.Layout.Row = 2;
	hCircButton.Layout.Column = 3;
	hCircButton.FontSize = 12;
	hCircButton.ButtonPushedFcn = @(btn, event) printButtonCallback(btn, event);
	
    % Create BW Selector Text
	hCircText = uilabel(sideCP_glayout);
	hCircText.Layout.Row = 3;
	hCircText.Layout.Column = 1;
	hCircText.Text = "Bandwidth Threshold:";
	hCircText.FontSize = 12;
	hCircText.HorizontalAlignment = 'right';
    
    % Create BW definition edit box
    hBWEdit = uieditfield(sideCP_glayout,'numeric', 'Limits', [-100 0], 'LowerLimitInclusive','on', 'UpperLimitInclusive','off', 'Value', optBW_param);
    hBWEdit.Layout.Row = 3;
    hBWEdit.Layout.Column = 2;
    hBWEdit.ValueChangedFcn = @(edt, event) changeBWDefnCallback(edt, event);
    
    sortByBW(optBW_param);
    
    % Create BW unit Text
	hCircText = uilabel(sideCP_glayout);
	hCircText.Layout.Row = 3;
	hCircText.Layout.Column = 3;
	hCircText.Text = " (dB)";
	hCircText.FontSize = 12;
	hCircText.HorizontalAlignment = 'left';
    
	% Initailize all axes (Subplots not used)
	ha_list = uiaxes(master_layout);
	ha_list.Layout.Row = 1;
	ha_list.Layout.Column = 1;
	for idx = 2:nplots
		ha_list(end+1) = uiaxes(master_layout);
		ha_list(end).Layout.Row    = ceil(idx./spc)+1;
		ha_list(end).Layout.Column = mod(idx-1, spc)+1;
	end

	% Initialize the UI.
	% Change units to normalized so components resize automatically.
	f.Units = 'normalized';
	ha.Units = 'normalized';
	hFreqText.Units = 'normalized';
	hFreqMenu.Units = 'normalized';
	hPwrText.Units = 'normalized';
	hPwrMenu.Units = 'normalized';

	% Plot initial data
% 	plot_page(solns, 1);

	% Assign the a name to appear in the window title.
% 	f.Name = 'Match Bandwidth UI';
	f.Name = 'Kinetic Inductance Up-Converter Analyzer';

	% Move the window to the center of the screen.
	movegui(f,'center')

	% Make the window visible.
	f.Visible = 'on';

	%  Pop-up menu callback. Read the pop-up menu Value property to
	%  determine which item is currently displayed and make it the
	%  current data. This callback automatically has access to 
	%  current_data because this function is nested at a lower level.
	function page_menu_callback(source,eventdata) 
		a=3;
% 		val = source.Value;
		
% 		% Determine the selected data set.
% 		strs = get(source, 'String');
% 		idx = get(source,'Value');

% 		page_no = str2num(val);
		
% 		plot_page(solns, page_no);
    end
	
    function changeBWDefnCallback(source, eventdata)
        a=3;
%         optBW_param = source.Value;
%         
%         % Recalculate BW and resort
%         sortByBW(optBW_param);
%         
%         % Refresh plots
%         plot_page(solns, page_no);
        
    end

    function sortByBW(bw)
        a=3;
%         % Calculate BW and add to list
%         BW_list = zeros(1, numel(solns));
%         idx = 1;
%         for s = solns
%             [BW_list(idx), ~, ~] = s.bandwidth("Absolute", bw);
%             idx = idx + 1;
%         end
% 
%         % Sort solutions by BW
%         [BW_list, I] = sort(BW_list, 'descend');
%         solns = solns(I);
        
    end

	function printButtonCallback(btn, event)
		a=3;
% 		% Get ID to search for
% 		find_id = str2num(hCircMenu.Value);
% 		
% 		found = false;
% 		
% 		% Search for ID
% 		for ss = solns
% 			if ss.ID == find_id
% 				if show_micro
% 					disp(ss.str("Circuit No. " + num2str(ss.ID), e_r, d));
% 				else
% 					disp(ss.str());
% 				end
% 				found = true;
% 			end
% 		end
% 		
% 		if ~found
% 			displ("Failed to find: ", find_id);
% 		end
		
	end

	function plot_page(ss, page_no)
		
		a = 3;
% 		for idx2 = 1:nplots
% 			
% 			% Check for out of bounds
% 			if (idx2+(page_no-1)*nplots > numel(ss))
% 				title(ha_list(idx2), " ");
% 				cla(ha_list(idx2));
% 				continue;
% 			end
% 			
% 			% Plot S-Parameter Response
% 			hold(ha_list(idx2), "off");
% 			plot(ha_list(idx2), freqs./1e9, lin2dB(abs(ss(idx2+(page_no-1)*nplots).G_in())));
% 			hold(ha_list(idx2), "on");
% 			
% 			% Calculate bandwidth
% 			[bw, f0, f1]=ss(idx2+(page_no-1)*nplots).bandwidth( "Absolute", optBW_param);
% 			
% 			% Add labels and title
% 			xlabel(ha_list(idx2), "Frequency (GHz)");
% 			ylabel(ha_list(idx2), "S-Parameter (dB)");
% 			plot_title = "No. " + num2str(ss(idx2+(page_no-1)*nplots).ID) + ", N = " + num2str(numel(ss(idx2+(page_no-1)*nplots).mats)/2) + ", [";
% 			for elmnt = ss(idx2+(page_no-1)*nplots).mats
% 				if elmnt.desc.type == "SHORT_STUB"
% 					plot_title = plot_title + "S";
% 				elseif elmnt.desc.type == "OPEN_STUB"
% 					plot_title = plot_title + "O";
% 				end
% 			end
% 			plot_title = plot_title + "], BW = " + num2fstr(bw/1e9) + " GHz";
% 			title(ha_list(idx2), plot_title);
% 			
% 			% Plot Bandwidth Region
% 			if ~isempty(bw)
% 				%fillregion([f0./1e9, f1./1e9], [NaN, NaN], [0, .8, 0], .2);
% 				line(ha_list(idx2), [f0./1e9, f0./1e9], y_bounds, 'Color', [.3, .3, .3], 'LineStyle', '--');
% 				line(ha_list(idx2), [f1./1e9, f1./1e9], y_bounds, 'Color', [.3, .3, .3], 'LineStyle', '--');
% 			else
% 				
% 			end
% 			
% 			% Add finishing touches to graph
% 			grid(ha_list(idx2), "on");
% 			ylim(ha_list(idx2), [-50, 0]);
% 			
% 		end

    end

    

end


























