function evalResults = evaluate_cnn(modelPath, prepParamsPath, splitInfoPath)
%EVALUATE_CNN Evaluates the trained CNN on held-out test hyperspectral images.
%
%   evalResults = evaluate_cnn()
%   evalResults = evaluate_cnn(modelPath, prepParamsPath, splitInfoPath)
%
% Computes:
%   - 4x4 Confusion Matrix
%   - Overall Accuracy
%   - Class-wise Precision, Recall, and F1-score
%   - Macro-averaged and Weighted Metrics
%   - Answers all Task 7 evaluation criteria

    if nargin < 1 || isempty(modelPath)
        modelPath = fullfile('Model', 'trained_cnn.mat');
    end
    if nargin < 2 || isempty(prepParamsPath)
        prepParamsPath = fullfile('Model', 'preprocessing_params.mat');
    end
    if nargin < 3 || isempty(splitInfoPath)
        splitInfoPath = fullfile('Model', 'split_info.json');
    end

    fprintf('=====================================================\n');
    fprintf('  Task 7: Final CNN Held-Out Test Evaluation\n');
    fprintf('=====================================================\n\n');

    % Load Model and Preprocessing Parameters
    if ~exist(modelPath, 'file')
        error('Model file not found: %s', modelPath);
    end
    if ~exist(prepParamsPath, 'file')
        error('Preprocessing params not found: %s', prepParamsPath);
    end

    cnnData = load(modelPath);
    prepData = load(prepParamsPath);

    classes = cellstr(prepData.classes);
    numClasses = numel(classes);
    selectedBands = prepData.selectedBandIndices_1based;
    targetSize = [prepData.targetHeight, prepData.targetWidth];

    % Read test files from split_info.json
    if exist(splitInfoPath, 'file')
        fid = fopen(splitInfoPath, 'r');
        rawJson = fread(fid, '*char')';
        fclose(fid);
        splitData = jsondecode(rawJson);
        testFiles = splitData.test_files;
        trueLabels = splitData.test_labels;
    else
        error('Split info JSON file not found: %s', splitInfoPath);
    end

    numTest = numel(testFiles);
    fprintf('Evaluating %d test images across %d polymer classes...\n', numTest, numClasses);
    if isfield(cnnData, 'trainedNet')
        fprintf('Deep Learning Engine: MATLAB Deep Learning Toolbox (DAGNetwork) Active\n');
    else
        fprintf('Deep Learning Engine: Native Matrix Forward Pass\n');
    end

    predLabels = cell(numTest, 1);
    predProbs  = zeros(numTest, numClasses);

    for i = 1:numTest
        bilFile = testFiles{i};
        [cube, ~] = load_hyperspectral_cube(bilFile);
        cube10 = cube(:, :, selectedBands);
        [paddedInput, ~] = pad_hyperspectral_image(cube10, targetSize, 0.0, 65535.0);

        if isfield(cnnData, 'trainedNet')
            try
                probs = predict(cnnData.trainedNet, single(paddedInput));
                [~, maxIdx] = max(probs);
                predClass = classes{maxIdx};
            catch
                [predClass, probs] = forward_cnn(cnnData, paddedInput);
            end
        else
            [predClass, probs] = forward_cnn(cnnData, paddedInput);
        end
        predLabels{i} = predClass;
        predProbs(i, :) = probs;
    end

    % 4x4 Confusion Matrix
    cm = zeros(numClasses, numClasses);
    for i = 1:numTest
        tIdx = find(strcmp(classes, trueLabels{i}));
        pIdx = find(strcmp(classes, predLabels{i}));
        cm(tIdx, pIdx) = cm(tIdx, pIdx) + 1;
    end

    totalSamples = sum(cm(:));
    overallAcc = sum(diag(cm)) / totalSamples;

    % Class-wise metrics
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    support = zeros(numClasses, 1);

    for c = 1:numClasses
        TP = cm(c, c);
        FP = sum(cm(:, c)) - TP;
        FN = sum(cm(c, :)) - TP;
        support(c) = TP + FN;

        if (TP + FP) > 0
            precision(c) = TP / (TP + FP);
        else
            precision(c) = 0.0; % Explicit zero denominator handling
        end

        if (TP + FN) > 0
            recall(c) = TP / (TP + FN);
        else
            recall(c) = 0.0;
        end

        if (precision(c) + recall(c)) > 0
            f1(c) = (2 * precision(c) * recall(c)) / (precision(c) + recall(c));
        else
            f1(c) = 0.0;
        end
    end

    macroPrec = mean(precision);
    macroRec = mean(recall);
    macroF1 = mean(f1);
    weightedF1 = sum(f1 .* support) / sum(support);

    % Display Results Table
    fprintf('\n----------------- 4x4 Confusion Matrix -----------------\n');
    fprintf('%10s | %8s %8s %8s %8s\n', 'True \\ Pred', classes{1}, classes{2}, classes{3}, classes{4});
    fprintf('-----------+------------------------------------\n');
    for r = 1:numClasses
        fprintf('%10s | %8d %8d %8d %8d\n', classes{r}, cm(r, 1), cm(r, 2), cm(r, 3), cm(r, 4));
    end
    fprintf('--------------------------------------------------------\n\n');

    fprintf('----------------- Class-wise Performance -----------------\n');
    fprintf('%8s | %10s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support');
    fprintf('---------+-----------------------------------------------\n');
    for c = 1:numClasses
        fprintf('%8s | %9.2f%% %9.2f%% %9.2f%% %10d\n', ...
            classes{c}, precision(c)*100, recall(c)*100, f1(c)*100, support(c));
    end
    fprintf('---------+-----------------------------------------------\n');
    fprintf('%8s | %9.2f%% %9.2f%% %9.2f%% %10d\n', ...
        'Macro Avg', macroPrec*100, macroRec*100, macroF1*100, totalSamples);
    fprintf('%8s | %9s %10s %9.2f%% %10d\n\n', ...
        'Weighted', '-', '-', weightedF1*100, totalSamples);

    fprintf('Overall Test Accuracy: %.2f%%\n\n', overallAcc * 100);

    evalResults = struct();
    evalResults.confusionMatrix = cm;
    evalResults.classes = classes;
    evalResults.overallAccuracy = overallAcc;
    evalResults.precision = precision;
    evalResults.recall = recall;
    evalResults.f1 = f1;
    evalResults.support = support;
    evalResults.macroPrecision = macroPrec;
    evalResults.macroRecall = macroRec;
    evalResults.macroF1 = macroF1;
    evalResults.weightedF1 = weightedF1;
end

function [predClass, probs] = forward_cnn(cnn, x)
% Pure MATLAB forward-pass implementation of the trained CNN
% x: HT x WT x 10 double array
    classes = cellstr(cnn.classes);

    % Block 1: Conv -> BN -> ReLU -> MaxPool
    x1 = conv2d_forward(x, cnn.conv1_weights, cnn.conv1_bias);
    x1 = bn_forward(x1, cnn.bn1_scale, cnn.bn1_offset, cnn.bn1_mean, cnn.bn1_var);
    x1 = max(0, x1);
    x1 = maxpool2d_forward(x1);

    % Block 2: Conv -> BN -> ReLU -> MaxPool
    x2 = conv2d_forward(x1, cnn.conv2_weights, cnn.conv2_bias);
    x2 = bn_forward(x2, cnn.bn2_scale, cnn.bn2_offset, cnn.bn2_mean, cnn.bn2_var);
    x2 = max(0, x2);
    x2 = maxpool2d_forward(x2);

    % Block 3: Conv -> BN -> ReLU -> MaxPool
    x3 = conv2d_forward(x2, cnn.conv3_weights, cnn.conv3_bias);
    x3 = bn_forward(x3, cnn.bn3_scale, cnn.bn3_offset, cnn.bn3_mean, cnn.bn3_var);
    x3 = max(0, x3);
    x3 = maxpool2d_forward(x3);

    % Global Average Pooling (over spatial H and W)
    gap = squeeze(mean(mean(x3, 1), 2)); % 64 x 1 vector

    % Fully Connected Layer: W * gap + b
    logits = cnn.fc_weights * gap(:) + cnn.fc_bias(:);

    % Softmax
    expLogits = exp(logits - max(logits));
    probs = (expLogits / sum(expLogits))';

    [~, maxIdx] = max(probs);
    predClass = classes{maxIdx};
end

function out = conv2d_forward(in, W, b)
% 2D Convolution with same padding
    [H, W_in, Cin] = size(in);
    [kH, kW, Cin_w, Cout] = size(W);
    padH = floor(kH / 2);
    padW = floor(kW / 2);
    padded = padarray(in, [padH, padW, 0], 0, 'both');

    out = zeros(H, W_in, Cout);
    for oc = 1:Cout
        acc = zeros(H, W_in);
        for ic = 1:Cin
            acc = acc + conv2(padded(:, :, ic), rot90(W(:, :, ic, oc), 2), 'valid');
        end
        out(:, :, oc) = acc + b(oc);
    end
end

function out = bn_forward(in, gamma, beta, mean_val, var_val)
% Batch Normalization inference: y = gamma * (x - mean) / sqrt(var + eps) + beta
    eps = 1e-5;
    [H, W, C] = size(in);
    out = zeros(H, W, C);
    for c = 1:C
        std_val = sqrt(var_val(c) + eps);
        out(:, :, c) = gamma(c) * ((in(:, :, c) - mean_val(c)) / std_val) + beta(c);
    end
end

function out = maxpool2d_forward(in)
% 2x2 Max Pooling with stride 2
    [H, W, C] = size(in);
    outH = floor(H / 2);
    outW = floor(W / 2);
    out = zeros(outH, outW, C);
    for c = 1:C
        for r = 1:outH
            for col = 1:outW
                rIdx = (r-1)*2 + (1:2);
                cIdx = (col-1)*2 + (1:2);
                out(r, col, c) = max(max(in(rIdx, cIdx, c)));
            end
        end
    end
end
