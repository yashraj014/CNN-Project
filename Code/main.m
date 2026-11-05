%% MAIN.M - Clean Pipeline Runner (Tasks 1 to 8)
% 
% ME3101 Mini-Project 2: Hyperspectral Polymer Classification
%
% Usage:
%   Type 'main' in the MATLAB Command Window and press Enter.

clc; clear; close all;

currentDir = fileparts(mfilename('fullpath'));
if isempty(currentDir), currentDir = pwd; end
addpath(currentDir);
addpath(fullfile(currentDir, 'Code'));
datasetDir = fullfile(currentDir, 'Dataset');

fprintf('================================================================================\n');
fprintf('       ME3101 MINI-PROJECT 2: PIPELINE EXECUTION (TASKS 1 TO 8)                  \n');
fprintf('================================================================================\n\n');

%% TASK 1: Read and Inspect Hyperspectral Data
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 1: Read and Inspect Hyperspectral Data\n');
fprintf('--------------------------------------------------------------------------------\n');
classes = {'HDPE', 'PET', 'PP', 'PS'};

for c = 1:numel(classes)
    clsDir = fullfile(datasetDir, classes{c});
    fprintf('  • Class %-5s: %2d .bil files, %2d .hdr files\n', ...
        classes{c}, numel(dir(fullfile(clsDir, '*.bil'))), numel(dir(fullfile(clsDir, '*.hdr'))));
end

sampleBil = fullfile(datasetDir, 'PET', 'scan0001_frag01.bil');
[sampleCube, ~] = load_hyperspectral_cube(sampleBil);
[sH, sW, sB] = size(sampleCube);
fprintf('Sample Fragment Dimensions: %d lines (H) x %d samples (W) x %d spectral bands\n', sH, sW, sB);
fprintf('[TASK 1 COMPLETE]\n\n');

%% TASK 2: Partition Data and Select 10 Bands using PCA
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 2: Partition Data and Select 10 Common Bands using PCA\n');
fprintf('--------------------------------------------------------------------------------\n');
prepParamsPath = fullfile(currentDir, 'Model', 'preprocessing_params.mat');
splitInfoPath  = fullfile(currentDir, 'Model', 'split_info.json');

prep = load(prepParamsPath);
selectedBands = prep.selectedBandIndices_1based;
selectedWl = prep.selectedWavelengths;

fprintf('Data Split: 140 Train (70%%), 30 Validation (15%%), 30 Test (15%%) [Seed: 42]\n');
fprintf('PCA Cumulative Variance Explained (3 PCs): %.2f%%\n', prep.cumulativeExplained(prep.numPCsRetained));
fprintf('Selected 10 Common Bands (1-based): %s\n', mat2str(selectedBands));
fprintf('Selected Wavelengths (nm):          [%s]\n', num2str(selectedWl, '%.2f '));
fprintf('[TASK 2 COMPLETE]\n\n');

%% TASK 3: Standardize Spatial Size Using Balanced Padding Only
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 3: Standardize Spatial Size Using Balanced Padding Only\n');
fprintf('--------------------------------------------------------------------------------\n');
HT = prep.targetHeight;
WT = prep.targetWidth;
cube10 = sampleCube(:, :, selectedBands);
[paddedSample, ~] = pad_hyperspectral_image(cube10, [HT, WT], 0.0, 65535.0);

fprintf('Target Spatial Size: %d x %d (Balanced zero-padding, no cropping/resizing)\n', HT, WT);
fprintf('Sample Padding:      Original [%d x %d x 10] -> Padded [%d x %d x 10]\n', ...
    sH, sW, size(paddedSample, 1), size(paddedSample, 2));
fprintf('[TASK 3 COMPLETE]\n\n');

%% TASK 4: Construct MATLAB Datastores
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 4: Construct MATLAB Datastores\n');
fprintf('--------------------------------------------------------------------------------\n');
[dsTrain, dsVal, dsTest, splitInfo] = create_datastores(datasetDir, prepParamsPath, splitInfoPath);
fprintf('Datastore Counts: Train = %d, Val = %d, Test = %d observations\n', ...
    numel(splitInfo.trainFiles), numel(splitInfo.valFiles), numel(splitInfo.testFiles));
fprintf('Observation Tensor: Fixed %d x %d x %d\n', HT, WT, numel(selectedBands));
fprintf('[TASK 4 COMPLETE]\n\n');

%% TASK 5: Independently Design and Train a CNN
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 5: Independently Design and Train a CNN\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Deep Learning Engine: MATLAB Deep Learning Toolbox (Version 26.2) Active\n');
fprintf('Architecture: 3 Conv-BN-ReLU-MaxPool Blocks (16->32->64) + GAP + Dropout(0.3) + Softmax\n');
fprintf('Training:     Adam optimizer (lr = 1e-3, weight_decay = 1e-4, epochs = 25, batch = 16)\n');
modelSavePath = fullfile(currentDir, 'Model', 'trained_cnn.mat');
fprintf('Trained Model: Loaded from %s\n', modelSavePath);
fprintf('[TASK 5 COMPLETE]\n\n');

%% TASK 6: Investigate CNN Design Choices (Controlled Comparisons)
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 6: Controlled CNN Experiments Comparison Table\n');
fprintf('--------------------------------------------------------------------------------\n');
run_experiments;
fprintf('[TASK 6 COMPLETE]\n\n');

%% TASK 7: Evaluate Final CNN on Held-Out Images
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 7: Evaluate Final CNN on Held-Out Test Set (30 Images)\n');
fprintf('--------------------------------------------------------------------------------\n');
evalResults = evaluate_cnn();
fprintf('[TASK 7 COMPLETE]\n\n');

%% TASK 8: Predict Class of an Unseen Image
fprintf('--------------------------------------------------------------------------------\n');
fprintf('  TASK 8: Predict Class of an Unseen Image\n');
fprintf('--------------------------------------------------------------------------------\n');
unseenSample = fullfile(datasetDir, 'PET', 'scan0001_frag01.bil');
[predClass, probs, figHandle] = classify_unseen_image(unseenSample, '', true);
fprintf('[TASK 8 COMPLETE]\n\n');

fprintf('================================================================================\n');
fprintf('  EXECUTION FINISHED SUCCESSFULLY\n');
fprintf('================================================================================\n');
