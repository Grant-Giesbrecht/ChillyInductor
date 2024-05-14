I_dc = 10e-3;
Q_ = 1/.19;
len = 98.8e-3;%0988;
L_ = 1e-6; % In Nathan's spreadsheet, he wrote 'L0' but plugged in (and meant) L'
Z0 = 90;

C_ = 123e-12;

% NOTE: This is dt as defined in Nathan's spreadsheet. It is NOT the time
% delay for the signal to reach the output; it is the difference in time
% delay between a signal propagating though the transmission line with zero
% nonlinearity versus with the nonlinearty.
nathan_dt = 0.5*I_dc^2*Q_^2*len*L_/Z0;
nathan_dt_ns = nathan_dt*1e9;

tot_dt = len*sqrt(L_*C_)*(1 + 0.5*I_dc^2*Q_^2);
tot_dt_ns = tot_dt*1e9;

T_10GHz_ns = 1/10e9*1e9;

num_periods = nathan_dt_ns/T_10GHz_ns;

displ("  Total time delay: ", tot_dt_ns, " ns");
displ("  Component from nonlinearity: ", nathan_dt_ns, " ns");