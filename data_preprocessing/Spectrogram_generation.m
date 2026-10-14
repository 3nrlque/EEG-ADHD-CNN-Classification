%% EEG Spectrogram Generation
% This script converts preprocessed EEG signal segments stored in MAT files
% into RGB spectrogram images for subsequent CNN-based classification.

clear all; close all; clc;

% Input and Output Directories
inFolder = '////'; 
outFolder = '////';

% Spectrogram Parameters
winSec = 2;             % Window length in seconds
overlapPercent = 0.90;  % Fraction of overlap between consecutive windows
freqMin = 0.5;          % Minimum frequency to retain (Hz)
freqMax = 40;           % Maximum frequency to retain (Hz)


fileList = dir(fullfile(inFolder, '**', '*.mat'));

% Process Each EEG Segment
for i = 1:numel(fileList)
     % Full path to the current MAT file
    filePath = fullfile(fileList(i).folder, fileList(i).name);
    load(filePath);
    
    % Convert the window length from seconds to samples
    winSamples = round(winSec * Fs); % 2s and 128Hz = 256 samples
    nOverlap = round(overlapPercent * winSamples); % 90% = 230 samples
    nfft = winSamples;
    
    % Initialize average power spectrogram
    P_sum = 0;
    numChannels = size(Fragment, 2);
    
    % Compute the power spectrogram independently for each channel
    for ch = 1:numChannels
        [~, f, ~, P] = spectrogram(Fragment(:, ch), hamming(winSamples), nOverlap, nfft, Fs);
        P_sum = P_sum + P; % Sum the power of all channels
    end
    
    P_avg = P_sum / numChannels; % Average power
    
    P_db = 10 * log10(P_avg); % logarithmic scale
    
    % Identify frequency bins within range
    idx_f = (f >= freqMin) & (f <= freqMax);
    P_db_cropped = P_db(idx_f, :);
    
     % Reverse frequency axis
    P_db_flip = flipud(P_db_cropped);

    % Normalize the spectrogram intensity values
    img_gray = mat2gray(P_db_flip);
    
    img_224 = imresize(img_gray, [224, 224]); % resize
    
    % Convert spectrogram to an RGB Image
    img_rgb = ind2rgb(gray2ind(img_224, 256), jet(256));
    
    % Save spectrogram Image
    outPathDir = strrep(fileList(i).folder, inFolder, outFolder);
    
    % Create output directory if it does't exist
    if ~exist(outPathDir, 'dir')
        mkdir(outPathDir);
    end
    
    % Save as PNG 
    [~, name, ~] = fileparts(fileList(i).name);
    outFileName = fullfile(outPathDir, strcat(name, '.png'));
    

    imwrite(img_rgb, outFileName);
    
end

fprintf('Completed\n');