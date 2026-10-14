%% Statistical Comparison of CNN Architectures
% This script compares the performance of five CNN architectures
% using the Friedman test and post-hoc pairwise comparisons
% with Bonferroni correction.
%
% Model order:
% 1. CNN-4
% 2. CNN-10
% 3. ResNet50V2
% 4. VGG16
% 5. InceptionV3

clc; clear;

% Load Performance Data from csv file
filename = "4cEC.csv";
data = readtable(filename);

X = table2array(data(:, 2:end));

% Define the model names in the same order as their columns in X
ModelNames = {'CNN-4', 'CNN-10', 'ResNet50V2', 'VGG16', 'InceptionV3'};

% Freidman test
[p, tbl, stats] = friedman(X, 1, 'off');

fprintf('Friedman p-value: %.6f\n', p);
stats.gnames = ModelNames';

% Post-hoc Pairwise Comparisons
% Perform post-hoc comparisons only if the Friedman test
% indicates a statistically significant difference between models
if p < 0.05

    figure;
    [c, m, h, gnames] = multcompare(stats, 'CType', 'bonferroni');

    title(['Model comparison: ' filename]);

    % Configure Plot Axis Labels
    ax = gca;

    % Set the y-axis tick labels to the model names
    ax.YTickLabel = ModelNames;

    % Set the x-axis tick labels to the model names
    ax.XTickLabel = ModelNames;

    % Display Pairwise Comparison Results
    disp('Multcompare Results:');

    % c = [i j diff CI_low CI_high p]
    for k = 1:size(c,1)
        fprintf('%s vs %s | p = %.6f\n', ...
            ModelNames{c(k,1)}, ...
            ModelNames{c(k,2)}, ...
            c(k,6));
    end

else
    disp('No statistically significant differences detected between CNN models');
end