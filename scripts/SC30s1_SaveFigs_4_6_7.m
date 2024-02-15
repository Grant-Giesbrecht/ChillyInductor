
clear CE2;
for idx_trace = numel(fig4):-1:1
	
	trace = fig4(idx_trace);
	
	ndata = struct();
	ndata.frequency_Hz = trace.XData.*1e9;
	ndata.CE2_pcnt = trace.YData;
	
	CE2(idx_trace) = 
	
end