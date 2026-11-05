function [predictedClass, classProbabilities, figureHandle] = classify_unseen_image(bilPath, hdrPath, showPlot)
%CLASSIFY_UNSEEN_IMAGE Predicts the polymer class of an unseen hyperspectral image.
%
%   predictedClass = classify_unseen_image(bilPath)
%   [predictedClass, classProbabilities, fig] = classify_unseen_image(bilPath, hdrPath, showPlot)
%
% TASK 8 COMPLIANT IMPLEMENTATION:
%   - Accepts any unseen .bil / .hdr hyperspectral image pair.
%   - Reuses saved common PCA band indices (does NOT refit PCA).
%   - Applies fixed balanced padding and normalization (does NOT retrain CNN).
%   - Detects if image dimensions exceed target dimensions [74, 74].
%   - Outputs predicted class ('HDPE', 'PET', 'PP', or 'PS') and class probability scores.
%   - Generates and saves demonstration figure showing pseudo-RGB and confidence scores.

    % Default test sample if no file is provided
    if nargin < 1 || isempty(bilPath)
        bilPath = fullfile('Dataset', 'PET', 'scan0001_frag01.bil');
        fprintf('No BIL path supplied. Using designated unseen demonstration image: %s\n', bilPath);
    end
    if nargin < 2 || isempty(hdrPath)
        [fdir, fname, ~] = fileparts(bilPath);
        hdrPath = fullfile(fdir, [fname, '.hdr']);
    end
    if nargin < 3 || isempty(showPlot)
        showPlot = true;
    end

    % Paths to saved parameters and model
    prepParamsPath = fullfile('Model', 'preprocessing_params.mat');
    modelPath = fullfile('Model', 'trained_cnn.mat');

    if ~exist(prepParamsPath, 'file')
        error('Preprocessing parameters file not found: %s', prepParamsPath);
    end
    if ~exist(modelPath, 'file')
        error('Trained CNN model not found: %s', modelPath);
    end

    % 1. Load Preprocessing Settings (Fixed PCA bands & Target Dimensions)
    prep = load(prepParamsPath);
    selectedBands = prep.selectedBandIndices_1based;
    targetSize = [prep.targetHeight, prep.targetWidth];
    classes = cellstr(prep.classes);
    normScale = prep.normScale;

    % 2. Load Hyperspectral Cube
    [cube, info] = load_hyperspectral_cube(bilPath, hdrPath);
    [H, W, B] = size(cube);

    fprintf('\n=== UNSEEN IMAGE INFERENCE (Task 8) ===\n');
    fprintf('Image Source:    %s\n', bilPath);
    fprintf('Header:          %s\n', hdrPath);
    fprintf('Raw Dimensions:  %d lines (H) x %d samples (W) x %d bands\n', H, W, B);

    % Detect and report if image exceeds established target dimensions
    if H > targetSize(1) || W > targetSize(2)
        error('Oversized input: Image dimensions [%d, %d] exceed target dimensions [%d, %d].', ...
            H, W, targetSize(1), targetSize(2));
    end

    % 3. Extract 10 common PCA bands
    cube10 = cube(:, :, selectedBands);

    % 4. Standardize spatial size using padding only (balanced constant zero)
    [paddedInput, padDetails] = pad_hyperspectral_image(cube10, targetSize, 0.0, normScale);
    fprintf('Padded Tensor:   %d x %d x %d (Balanced zero-padding: Top=%d, Bot=%d, Left=%d, Right=%d)\n', ...
        targetSize(1), targetSize(2), numel(selectedBands), ...
        padDetails.padTop, padDetails.padBottom, padDetails.padLeft, padDetails.padRight);

    % 5. Load Trained CNN and Predict
    cnn = load(modelPath);

    % Check if standard MATLAB Deep Learning Toolbox is present
    if isfield(cnn, 'trainedNet')
        fprintf('Deep Learning Engine: MATLAB Deep Learning Toolbox (DAGNetwork) Active\n');
        try
            probs = predict(cnn.trainedNet, single(paddedInput));
            [~, maxIdx] = max(probs);
            predictedClass = classes{maxIdx};
        catch ME
            if isfield(cnn, 'conv1_weights')
                fprintf('Falling back to native matrix forward pass...\n');
                [predictedClass, probs] = forward_cnn(cnn, paddedInput);
            else
                rethrow(ME);
            end
        end
    else
        fprintf('Deep Learning Engine: Native Matrix Forward Pass\n');
        [predictedClass, probs] = forward_cnn(cnn, paddedInput);
    end

    classProbabilities = probs(:)';

    fprintf('\n---------------- INFERENCE RESULTS ----------------\n');
    fprintf('PREDICTED CLASS:  >> %s <<\n', predictedClass);
    fprintf('Class Probabilities:\n');
    for c = 1:numel(classes)
        fprintf('  %-6s: %6.2f%%\n', classes{c}, classProbabilities(c) * 100);
    end
    fprintf('---------------------------------------------------\n\n');

    % 6. Visualization Demonstration
    figureHandle = figure('Name', 'Unseen Hyperspectral Classification', ...
                          'Color', 'w', 'Position', [100, 100, 950, 420], 'Visible', 'off');

    % Subplot 1: Pseudo-RGB Visualization
    subplot(1, 2, 1);
    % Pseudo-RGB from 3 selected bands (e.g. bands 1, 5, 10 of the 10 selected bands)
    r = paddedInput(:, :, 1);
    g = paddedInput(:, :, 5);
    b = paddedInput(:, :, 10);
    rNorm = (r - min(r(:))) / (max(r(:)) - min(r(:)) + 1e-6);
    gNorm = (g - min(g(:))) / (max(g(:)) - min(g(:)) + 1e-6);
    bNorm = (b - min(b(:))) / (max(b(:)) - min(b(:)) + 1e-6);
    pseudoRGB = cat(3, rNorm, gNorm, bNorm);

    imshow(pseudoRGB);
    [~, fname, fext] = fileparts(bilPath);
    title(sprintf('Unseen Input: %s%s\nRaw: %dx%d | Padded: %dx%d', fname, fext, H, W, targetSize(1), targetSize(2)), ...
        'FontSize', 11, 'FontWeight', 'bold');

    % Subplot 2: Prediction Scores Bar Chart
    subplot(1, 2, 2);
    barColors = [0.12, 0.47, 0.71;  % HDPE
                 1.00, 0.50, 0.05;  % PET
                 0.17, 0.63, 0.17;  % PP
                 0.84, 0.15, 0.16]; % PS
    bPlot = bar(1:numel(classes), classProbabilities * 100, 'FaceColor', 'flat');
    bPlot.CData = barColors;
    set(gca, 'XTick', 1:numel(classes), 'XTickLabel', classes, 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Classification Probability (%)', 'FontSize', 11, 'FontWeight', 'bold');
    ylim([0, 110]);
    grid on;
    title(sprintf('Predicted Polymer: %s (%.1f%%)', predictedClass, max(classProbabilities) * 100), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.1, 0.4, 0.1]);

    for c = 1:numel(classes)
        val = classProbabilities(c) * 100;
        text(c, val + 2.5, sprintf('%.1f%%', val), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end

    % Save Demonstration Figure
    plotDir = fullfile('Results', 'Plots');
    if ~exist(plotDir, 'dir'), mkdir(plotDir); end
    saveas(figureHandle, fullfile(plotDir, 'unseen_image_prediction_matlab.png'));
    fprintf('Demonstration plot saved to: %s\n', fullfile(plotDir, 'unseen_image_prediction_matlab.png'));

    if showPlot
        set(figureHandle, 'Visible', 'on');
    end
end

function [predClass, probs] = forward_cnn(cnn, x)
% Pure MATLAB forward-pass implementation of the trained CNN
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

    % Global Average Pooling
    gap = squeeze(mean(mean(x3, 1), 2));

    % Fully Connected Layer
    logits = cnn.fc_weights * gap(:) + cnn.fc_bias(:);

    % Softmax
    expLogits = exp(logits - max(logits));
    probs = (expLogits / sum(expLogits))';

    [~, maxIdx] = max(probs);
    predClass = classes{maxIdx};
end

function out = conv2d_forward(in, W, b)
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
    eps = 1e-5;
    [H, W, C] = size(in);
    out = zeros(H, W, C);
    for c = 1:C
        std_val = sqrt(var_val(c) + eps);
        out(:, :, c) = gamma(c) * ((in(:, :, c) - mean_val(c)) / std_val) + beta(c);
    end
end

function out = maxpool2d_forward(in)
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
