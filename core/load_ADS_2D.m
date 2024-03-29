function [freq_HB, freq_SG, Vbias, Pload_mag, Pload_deg] = importfile(filename, dataLines)
%IMPORTFILE Import data from a text file
%  [FREQ_HB, FREQ_SG, VBIAS, PLOAD_MAG, PLOAD_DEG] =
%  IMPORTFILE(FILENAME) reads data from text file FILENAME for the
%  default selection.  Returns the data as column vectors.
%
%  [FREQ_HB, FREQ_SG, VBIAS, PLOAD_MAG, PLOAD_DEG] = IMPORTFILE(FILE,
%  DATALINES) reads data for the specified row interval(s) of text file
%  FILENAME. Specify DATALINES as a positive scalar integer or a N-by-2
%  array of positive scalar integers for dis-contiguous row intervals.
%
%  Example:
%  [freq_HB, freq_SG, Vbias, Pload_mag, Pload_deg] = importfile("/Users/grantgiesbrecht/Downloads/coarse_100K_2Dsweep.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 18-Oct-2023 19:12:26

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
	dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 5);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = [",", "/"];

% Specify column names and types
opts.VariableNames = ["freq_HB", "freq_SG", "Vbias", "Pload_mag", "Pload_deg"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "freq_HB", "TrimNonNumeric", true);
opts = setvaropts(opts, "freq_HB", "ThousandsSeparator", ",");

% Import the data
tbl = readtable(filename, opts);

%% Convert to output type
freq_HB = tbl.freq_HB';
freq_SG = tbl.freq_SG';
Vbias = tbl.Vbias';
Pload_mag = tbl.Pload_mag';
Pload_deg = tbl.Pload_deg';
end