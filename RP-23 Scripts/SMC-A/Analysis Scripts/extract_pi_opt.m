close all;

fig_doub = open("G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\doubler\halfpi opt\fit.fig");
fig_trad = open("G:\ARC0 PhD Data\RP-23 Qubit Readout\Data\SMC-A\Misc\Mai_Exodus\grant_data_mai\SF10ns\trad\halfpi opt\fit.fig");


h_doub = findobj(fig_doub,'Type','line');
doub_x = get(h_doub,'XData');
doub_y = get(h_doub,'YData');
disp('X Data:');
disp(doub_x);
disp('Y Data:');
disp(doub_y);

h_trad = findobj(fig_trad,'Type','line');
trad_x = get(h_trad,'XData');
trad_y = get(h_trad,'YData');
disp('X Data:');
disp(trad_x);
disp('Y Data:');
disp(trad_y);

fig1 = figure(10);
hold off;
plot(doub_x{2}+0.015, doub_y{2}-mean(doub_y{2}));
hold on;
plot(trad_x{2}, trad_y{2}-mean(trad_y{2}));

%% Print Python friendly lists

s = "doub_x = " + python_list(doub_x{2}) + fstr("\n");
s = s + "doub_y = " + python_list(doub_y{2}) + fstr("\n");
s = s + "trad_x = " + python_list(trad_x{2}) + fstr("\n");
s = s + "trad_y = " + python_list(trad_y{2}) + fstr("\n") + fstr("\n");

s = s + fstr('\n') + "doub_x_fit = " + python_list(doub_x{1}) + fstr("\n");
s = s + "doub_y_fit = " + python_list(doub_y{1}) + fstr("\n");
s = s + "trad_x_fit = " + python_list(trad_x{1}) + fstr("\n");
s = s + "trad_y_fit = " + python_list(trad_y{1}) + fstr("\n") + fstr("\n");

displ(s);
clipboard('copy', s);

