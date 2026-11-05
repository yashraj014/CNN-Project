%% TRAIN_CNN Main MATLAB training script for Hyperspectral Polymer Classification
%
% This script:
%   1. Loads the preprocessing parameters (10 PCA selected bands).
%   2. Pre-loads and standardizes training and validation datasets.
%   3. Constructs the 3-block 2D CNN architecture using MATLAB Deep Learning Toolbox.
%   4. Trains the network using Adam optimizer and saves the trained DAGNetwork into Model/trained_cnn.mat.

clc; clear; close all;

fprintf('========================================================================\n');
fprintf('  ME3101: Native MATLAB CNN Training (Deep Learning Toolbox)\n');
fprintf('========================================================================\n\n');

% Paths
currentDir = fileparts(mfilename('fullpath'));
if isempty(currentDir), currentDir = pwd; end
rootDir = fullfile(currentDir, '..');
if ~exist(fullfile(rootDir, 'Dataset'), 'dir')
    rootDir = currentDir;
end

datasetDir = fullfile(rootDir, 'Dataset');
prepParamsPath = fullfile(rootDir, 'Model', 'preprocessing_params.mat');
splitInfoPath  = fullfile(rootDir, 'Model', 'split_info.json');
modelSavePath  = fullfile(rootDir, 'Model', 'trained_cnn.mat');

% 1. Load Preprocessing Parameters
if ~exist(prepParamsPath, 'file')
    error('Preprocessing parameters file not found at: %s', prepParamsPath);
end
prep = load(prepParamsPath);
selectedBands = prep.selectedBandIndices_1based;
targetSize = [prep.targetHeight, prep.targetWidth];
classes = cellstr(prep.classes);
numClasses = numel(classes);

fprintf('Target input tensor size: %d x %d x %d\n', targetSize(1), targetSize(2), numel(selectedBands));
fprintf('Selected PCA bands:       %s\n', mat2str(selectedBands));
fprintf('Target polymer classes:   %s\n\n', strjoin(classes, ', '));

% 2. Read Split Files
if ~exist(splitInfoPath, 'file')
    error('Split info file not found at: %s', splitInfoPath);
end
fid = fopen(splitInfoPath, 'r');
splitData = jsondecode(fread(fid, '*char')');
fclose(fid);

trainFiles = splitData.train_files;
trainLabels = categorical(splitData.train_labels, classes);
valFiles = splitData.val_files;
valLabels = categorical(splitData.val_labels, classes);

numTrain = numel(trainFiles);
numVal = numel(valFiles);

fprintf('Loading and standardizing %d training and %d validation cubes...\n', numTrain, numVal);

XTrain = zeros(targetSize(1), targetSize(2), numel(selectedBands), numTrain, 'single');
for i = 1:numTrain
    fpath = trainFiles{i};
    if ~exist(fpath, 'file')
        [~, fname, fext] = fileparts(fpath);
        fpath = fullfile(datasetDir, char(trainLabels(i)), [fname, fext]);
    end
    cube = load_hyperspectral_cube(fpath);
    cube10 = cube(:, :, selectedBands);
    [padded, ~] = pad_hyperspectral_image(cube10, targetSize, 0.0, 65535.0);
    XTrain(:, :, :, i) = single(padded);
end

XVal = zeros(targetSize(1), targetSize(2), numel(selectedBands), numVal, 'single');
for i = 1:numVal
    fpath = valFiles{i};
    if ~exist(fpath, 'file')
        [~, fname, fext] = fileparts(fpath);
        fpath = fullfile(datasetDir, char(valLabels(i)), [fname, fext]);
    end
    cube = load_hyperspectral_cube(fpath);
    cube10 = cube(:, :, selectedBands);
    [padded, ~] = pad_hyperspectral_image(cube10, targetSize, 0.0, 65535.0);
    XVal(:, :, :, i) = single(padded);
end

fprintf('Data loaded successfully. Tensor size: [%s]\n\n', num2str(size(XTrain)));

% 3. Build 2D CNN Architecture (Baseline 3-Block Architecture)
fprintf('Building 2D CNN Architecture with Batch Normalization...\n');
lgraph = build_cnn_model('medium', [targetSize, numel(selectedBands)], numClasses, 0.3);

% 4. Training Options
epochs = 25;
miniBatchSize = 16;
initialLearnRate = 1e-3;
weightDecay = 1e-4;

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'L2Regularization', weightDecay, ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, valLabels}, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'none');

% 5. Train Network
fprintf('\nStarting training with Adam optimizer...\n');
trainedNet = trainNetwork(XTrain, trainLabels, lgraph, options);

% 6. Save Model
% Save the trained DAGNetwork object
save(modelSavePath, 'trainedNet', 'classes', 'targetSize', 'selectedBands');
fprintf('\n[SUCCESS] CNN training complete. Native trained network saved to:\n  %s\n', modelSavePath);
