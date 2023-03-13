
%%% Loads ECG signals and labels from .csv file %%%

% ECG raw signals should be previously stored and ready %
% to load in parent directory Subjects. %

% .csv file contains the filename associated to each subject and its label (userX)
% Se crea una tabla con los nombres de los ficheros de los sujetos y su clase

ref = 'dataset.csv';
tbl = readtable(ref,'ReadVariableNames',false);
tbl.Properties.VariableNames = {'Filename','Class'};

% Each file is loaded and its ECG signal saved in table
for i = 1:height(tbl)
    fileData = load(['./Sujetos/',[tbl.Filename{i},'.mat']]);
    % Discard second ECG channel of our signal for the study 
    tbl.Signal{i} = fileData.val(1,:); 
end

% Data preparation for ingestion in LSTM format
% Signals: cell array 1xn of each subject
% Labels: categorical array of each subject
Signals = tbl.Signal;
Classes = categorical(tbl.Class);

% Save these vars in .mat files
save ECGSignalsData.mat Signals Classes


