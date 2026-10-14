%% EEG Signal Segmentation
% This script divides preprocessed EEG recordings into fixed-length
% segments and saves the resulting data for subsequent analysis

clc; clear all; close all; 

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



% Configure Input Data
% Root directory containing the preprocessed EEG recordings
rootFolder = "";
filereadpath=rootFolder;

subjects=subjectsH;

% save(strcat(Fileread,filename), 'EEG_clipped','EEG_scaled', 'EEG_LAP', 'EEG_ICA', 'sources_ICA', 'EEG_resampled', 'time_new','Fs_new', 'commonCh');

% EEG channel labels
 chan_labels = {'Fp1','Fp2','F7','F3','Fz','F4', 'F8', 'T3', 'C3', 'Cz', 'C4','T4', 'T5', 'P3', 'Pz','P4','T6', 'O1','O2'}; 
   
% Process Each Participant
for i = 1:numel(subjects)
    % Get the current participant ID
    subjID = subjects{i};
    list_filename=strcat('EO_segments_',subjID);

    % Construct the path to the participant's recording
    eegFolder = fullfile(rootFolder, subjID, 'ses-2', list_filename);
    load(eegFolder)

    %EEG_clipped; 

         %% Signal segmentation
            % EEG_resampled = EEG_clipped;
            % Split the EEG signal into 10-second segments with a 1-second step
            
            % Select EEG Channels
            selected_channels = [4, 5, 6, 13];
            EEG_data = EEG_clipped(:, selected_channels);

            time = time_new';

            % Configure Segmentation Parameters
            winTime = 10;                % Segment duration in seconds
            stepTime = 10;               % Step between segments in seconds
            Fs = Fs_new;                 % Sampling frequency

            win_length = winTime * Fs;  % number of samples
            step_length = stepTime * Fs;

            [num_samples, num_channels] = size(EEG_data);

            % Calculate number of complete segments
            num_segments = floor((num_samples - win_length) / step_length) + 1;

            % Structure initialization
            segments(num_segments) = struct('data', [], 'time', []);

            % Divide the EEG Recording into Segments
            for k = 1:num_segments
                idx_start = (k-1)*step_length + 1;
                idx_end   = idx_start + win_length - 1;

                % Extract the EEG samples for the current segment
                segment_data = EEG_data(idx_start:idx_end, :);

                % Extract the time values corresponding to the same segment
                segment_time = time(idx_start:idx_end, :);

                % Store the EEG data and time vector in the structure array
                segments(k).data = segment_data;
                segments(k).time = segment_time;
            end


            %% Configure the Output Directory
            oldPath=filereadpath;
            newPath = strrep(oldPath, 'New Preprocessed LAP ICA 26 Channels', '10s SEGMENTS 4 kanały');
            filename=erase(list_filename,strcat('_',subjID));
            Fileread=strcat(newPath,subjID,'\');

            disp(newPath)

            % Create the participant's output directory if it does not exist
            if ~exist(Fileread, 'dir')
                mkdir(Fileread)
            end

            %% Save the Segmented EEG Data 
            Fileread=strcat(newPath,subjID,'\',filename,'.mat');
            signal=[EEG_clipped,time_new'];
            save(Fileread, 'signal', 'segments','Fs_new');
 

end

 fprintf('EEG segmentation completed\n');