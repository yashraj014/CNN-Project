function [dsTrain, dsVal, dsTest, splitInfo] = create_datastores(datasetRoot, prepParamsPath, splitInfoPath)
%CREATE_DATASTORES Builds MATLAB datastores for training, validation, and test splits.
%
%   [dsTrain, dsVal, dsTest, splitInfo] = create_datastores(datasetRoot)
%   [dsTrain, dsVal, dsTest, splitInfo] = create_datastores(datasetRoot, prepParamsPath, splitInfoPath)
%
% INPUTS:
%   datasetRoot    - Root directory containing class subfolders (HDPE, PET, PP, PS).
%   prepParamsPath - Path to Model/preprocessing_params.mat.
%   splitInfoPath  - (Optional) Path to Model/split_info.json containing precomputed splits.
%
% OUTPUTS:
%   dsTrain   - CombinedDatastore or custom datastore yielding [HT x WT x 10] and categorical label.
%   dsVal     - Datastore for validation set.
%   dsTest    - Datastore for test set.
%   splitInfo - Struct containing file lists and labels for each split.

    if nargin < 1 || isempty(datasetRoot)
        datasetRoot = 'Dataset';
    end
    if nargin < 2 || isempty(prepParamsPath)
        prepParamsPath = fullfile('Model', 'preprocessing_params.mat');
    end

    % Load preprocessing parameters
    if ~exist(prepParamsPath, 'file')
        error('Preprocessing parameters file not found: %s', prepParamsPath);
    end
    prep = load(prepParamsPath);
    selectedBands = prep.selectedBandIndices_1based;
    targetSize = [prep.targetHeight, prep.targetWidth];
    classes = cellstr(prep.classes);

    % Load split info if available for exact reproducible partitioning
    if nargin >= 3 && ~isempty(splitInfoPath) && exist(splitInfoPath, 'file')
        fid = fopen(splitInfoPath, 'r');
        rawJson = fread(fid, '*char')';
        fclose(fid);
        splitData = jsondecode(rawJson);
        trainFiles = splitData.train_files;
        trainLabels = splitData.train_labels;
        valFiles = splitData.val_files;
        valLabels = splitData.val_labels;
        testFiles = splitData.test_files;
        testLabels = splitData.test_labels;
    else
        % Scan datasetRoot and create stratified 70/15/15 split
        rng(42);
        trainFiles = {}; trainLabels = {};
        valFiles = {};   valLabels = {};
        testFiles = {};  testLabels = {};

        for c = 1:numel(classes)
            clsName = classes{c};
            clsDir = fullfile(datasetRoot, clsName);
            bilList = dir(fullfile(clsDir, '*.bil'));
            numFiles = numel(bilList);
            shuffIdx = randperm(numFiles);

            nTrain = 35;
            if ismember(clsName, {'HDPE', 'PP'})
                nVal = 7;
            else
                nVal = 8;
            end
            nTest = numFiles - nTrain - nVal;

            trainIdx = shuffIdx(1:nTrain);
            valIdx   = shuffIdx(nTrain+1 : nTrain+nVal);
            testIdx  = shuffIdx(nTrain+nVal+1 : end);

            for i = trainIdx
                trainFiles{end+1} = fullfile(clsDir, bilList(i).name); %#ok<AGROW>
                trainLabels{end+1} = clsName; %#ok<AGROW>
            end
            for i = valIdx
                valFiles{end+1} = fullfile(clsDir, bilList(i).name); %#ok<AGROW>
                valLabels{end+1} = clsName; %#ok<AGROW>
            end
            for i = testIdx
                testFiles{end+1} = fullfile(clsDir, bilList(i).name); %#ok<AGROW>
                testLabels{end+1} = clsName; %#ok<AGROW>
            end
        end
    end

    splitInfo = struct();
    splitInfo.trainFiles = trainFiles(:);
    splitInfo.trainLabels = categorical(trainLabels(:), classes);
    splitInfo.valFiles = valFiles(:);
    splitInfo.valLabels = categorical(valLabels(:), classes);
    splitInfo.testFiles = testFiles(:);
    splitInfo.testLabels = categorical(testLabels(:), classes);

    % Custom reader function handle
    readerFunc = @(filepath) custom_hyperspectral_reader(filepath, selectedBands, targetSize);

    % Build Datastores
    dsTrain = create_split_datastore(splitInfo.trainFiles, splitInfo.trainLabels, readerFunc);
    dsVal   = create_split_datastore(splitInfo.valFiles,   splitInfo.valLabels,   readerFunc);
    dsTest  = create_split_datastore(splitInfo.testFiles,  splitInfo.testLabels,  readerFunc);

    fprintf('--- MATLAB Datastores Initialized ---\n');
    fprintf('Training Datastore:   %d observations\n', numel(splitInfo.trainFiles));
    fprintf('Validation Datastore: %d observations\n', numel(splitInfo.valFiles));
    fprintf('Test Datastore:       %d observations\n', numel(splitInfo.testFiles));
    fprintf('Observation Tensor Size: %d x %d x %d\n', targetSize(1), targetSize(2), numel(selectedBands));
    disp('Class distribution in training set:');
    disp(countcats(splitInfo.trainLabels));
end

function data = custom_hyperspectral_reader(filepath, selectedBands, targetSize)
% Reads BIL, extracts 10 bands, normalizes, and applies balanced padding
    cube = load_hyperspectral_cube(filepath);
    cube10 = cube(:, :, selectedBands);
    [data, ~] = pad_hyperspectral_image(cube10, targetSize, 0.0, 65535.0);
end

function dsCombined = create_split_datastore(fileList, labels, readerFunc)
% Combines image datastore with categorical labels
    if exist('imageDatastore', 'file')
        dsImgs = imageDatastore(fileList, 'FileExtensions', {'.bil'}, 'ReadFcn', readerFunc);
        dsLabels = arrayDatastore(labels);
        dsCombined = combine(dsImgs, dsLabels);
    else
        % Struct wrapper when Deep Learning Toolbox is not installed
        dsCombined = struct('Files', {fileList}, 'Labels', labels, 'ReadFcn', readerFunc);
    end
end
