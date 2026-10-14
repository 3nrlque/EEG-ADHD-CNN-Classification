%% Save segmented eeg data
% This script extracts individual EEG segments from preprocessed recordings
% and saves each segment as a separate MAT-file.

clear all; close all; clc; 

% 1 - ADHD 
% 0 - Control 

% Root directory containing the segmented EEG recordings
 rootFolder = 'C:\Users\Enrique\Desktop\Studia\magisterka\praca magisterska\10s SEGMENTS 4 kanały';

% Define Participant Groups
% Participant IDs for adult healthy control subjects
subjectsH = { "sub-87974665", "sub-87974709", "sub-87974841", "sub-87974973", "sub-87976193", "sub-87976369", "sub-87976457", ...
"sub-87976505", "sub-87976817", "sub-87976953", "sub-87977045", "sub-87980241", "sub-87980373", "sub-87980417", ...
"sub-87980869", "sub-87982225", "sub-87982849", "sub-88008997", "sub-88041893", "sub-88041941", "sub-88048729", ...
"sub-88049585", "sub-88051073", "sub-88053453", "sub-88053545", "sub-88055121", "sub-88055301", "sub-88057461", ...
"sub-88057869", "sub-88058001", "sub-88058633", "sub-88059397", "sub-88067357", "sub-88067853", "sub-88068841", ...
"sub-88073029", "sub-88075053"};

% Participant IDs for adult subjects diagnosed with ADHD.
subjectsA = { "sub-88015117", "sub-88015565", "sub-88028433", "sub-88029425", "sub-88029557", "sub-88037801", "sub-88038789"...
"sub-88040317", "sub-88040853", "sub-88041081", "sub-88041257", "sub-88042705", "sub-88046753", "sub-88051337"...
"sub-88052645", "sub-88053137", "sub-88057101", "sub-88057781", "sub-88058589", "sub-88059305", "sub-88060609"...
"sub-88060833", "sub-88061325", "sub-88061829", "sub-88062189", "sub-88063217", "sub-88064433", "sub-88064477"...
"sub-88065021", "sub-88065921", "sub-88066909", "sub-88067449", "sub-88067493", "sub-88067541", "sub-88067717"...
"sub-88069289", "sub-88071453", "sub-88072037", "sub-88074021", "sub-88076989"};

% Sampling frequency after preprocessing (Hz)
Fs=128;

subjects=subjectsA;

% Channel labels for the 19-channel EEG configuration
chan_19 = {'Fp1','Fp2','F7','F3','Fz','F4', 'F8', 'T3', 'C3', 'Cz', 'C4','T4', 'T5', 'P3', 'Pz','P4','T6', 'O1','O2'}; 

% Output directory for individual 10-second EEG segments
rootOutFolder = fullfile(rootFolder,'Fragmenty_10s_4_kanały'); % <--- nowy folder na fragmenty

% Process each participant in the selected group
for i = 1:numel(subjects)
    subjID = subjects{i};
    
    % Locate the directory containing this participant's EEG files
    eegFolder = fullfile(rootFolder, 'ADHD', subjID); 

    % Find all MAT-files in the participant's directory
    fileList = dir(fullfile(eegFolder, '*.mat'));

    % Process each EEG recording
    for f = 1:numel(fileList)

        % Identify recording condition:
        % EO - eyes open
        % EC - eyes closed
        if contains(fileList(f).name, "EO")
            sessionLabel = 'EO';
        elseif contains(fileList(f).name, "EC")
            sessionLabel = 'EC';
        else
            continue; 
        end

        % Construct the full path to the input file
        filePath = fullfile(fileList(f).folder, fileList(f).name);
        fprintf('Reading: %s (Condition: %s)\n', filePath, sessionLabel);

        % Load the MAT-file containing the segmented EEG recording
        load(filePath)

        % Retrieve the sampling frequency
        if exist('Fs_new', 'var')
            Fs = Fs_new;
        end

        % Assign the class label based on the input file path
        % 0 - Control
        % 1 - ADHD
        filePathLower = lower(filePath);
        if contains(filePathLower, 'control')
            classLabel = 0;
        else 
            classLabel = 1;
        end

        % Create the output directory structure
        pathClass = fullfile(rootOutFolder, num2str(classLabel));
        pathSession = fullfile(pathClass, sessionLabel);
        subjOutFolder = fullfile(pathSession, subjID);

        % Create the participant's output directory if it does not exist
        if ~exist(subjOutFolder,'dir')
            mkdir(subjOutFolder)
        end

        % Extract and save each EEG segment as a separate MAT-file
        for fi = 1:length(segments)
            Fragment = segments(fi).data;
            T_Fragment = segments(fi).time;

            % Generate a unique filename for the segment
            filename = sprintf('Fragment_%03d.mat', fi);
            
            % Save the segment data, time vector, and sampling frequency
            save(fullfile(subjOutFolder, filename), 'Fragment', 'T_Fragment', 'Fs');
        end

    end
end
