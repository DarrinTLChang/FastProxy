clc;
clear;
close all;

% Folder containing the ADC .mat files
input_folder = 'D:\s531\processed data from 531\Mat Data\E\CL testing\period2';

% Find all ADC mat files in the folder
files = dir(fullfile(input_folder, 'ADC*.mat'));

% Loop through each file
for k = 1:length(files)
    file_path = fullfile(input_folder, files(k).name);

    % Load the .mat file
    S = load(file_path);

    % Check that required variables exist
    if ~isfield(S, 'data') || ~isfield(S, 'fs')
        warning('Skipping %s because it does not contain both "data" and "fs".', files(k).name);
        continue;
    end

    data = S.data(:);   % force column vector
    fs   = S.fs;

    % Create time vector in seconds
    N = length(data);
    time_s = (0:N-1)' / fs;

    % Create table
    T = table(time_s, data, 'VariableNames', {'time_s', 'value'});

    % Output file names
    [~, base_name, ~] = fileparts(files(k).name);
    csv_name  = fullfile(input_folder, [base_name, '.csv']);

    % Write spreadsheet files
    writetable(T, csv_name);

    fprintf('Saved: %s', csv_name);
end

disp('Done converting all ADC files.');