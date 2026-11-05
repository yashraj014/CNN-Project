function [selectedBands, importance, explained, cumExplained, numPCs] = select_pca_bands(X_train, numBands, varianceThreshold)
%SELECT_PCA_BANDS Selects 10 common original spectral bands using PCA on training data only.
%
%   [selectedBands, importance, explained, cumExplained, numPCs] = select_pca_bands(X_train)
%   [selectedBands, importance, explained, cumExplained, numPCs] = select_pca_bands(X_train, numBands, varianceThreshold)
%
% INPUTS:
%   X_train           - N x B numeric array (N training pixel spectra, B=168 bands).
%   numBands          - Number of original bands to retain (default: 10).
%   varianceThreshold - Cumulative explained variance threshold in percent (default: 95.0).
%
% OUTPUTS:
%   selectedBands     - 1 x numBands vector of 1-based original band indices (sorted ascending).
%   importance        - B x 1 vector of variance-weighted squared PCA loading scores.
%   explained         - B x 1 vector of percentage variance explained by each PC.
%   cumExplained      - B x 1 vector of cumulative variance explained.
%   numPCs            - Number of PCs required to reach the varianceThreshold.
%
% This function operates on training data only and works identically with both MATLAB's
% Statistics and Machine Learning Toolbox (pca) and native SVD/eigendecomposition.

    if nargin < 2 || isempty(numBands)
        numBands = 10;
    end
    if nargin < 3 || isempty(varianceThreshold)
        varianceThreshold = 95.0;
    end

    [N, B] = size(X_train);
    if N < 2
        error('At least two training spectra are required to perform PCA.');
    end

    % Centered PCA: subtract band-wise mean
    mu = mean(X_train, 1);
    X_centered = X_train - mu;

    % Check if Statistics Toolbox 'pca' is available
    if exist('pca', 'file') || exist('pca', 'builtin')
        try
            [coeff, ~, ~, ~, explained] = pca(X_train, 'Centered', true, 'Economy', true);
        catch
            [coeff, explained] = native_pca(X_centered, N, B);
        end
    else
        [coeff, explained] = native_pca(X_centered, N, B);
    end

    cumExplained = cumsum(explained);
    numPCs = find(cumExplained >= varianceThreshold - 1e-10, 1, 'first');
    if isempty(numPCs)
        numPCs = numel(explained);
    end

    % Variance-weighted squared loadings across retained PCs
    weights = explained(1:numPCs) ./ sum(explained(1:numPCs));
    importance = (coeff(:, 1:numPCs).^2) * weights;

    % Rank bands by importance and sort the top numBands in ascending spectral order
    [~, ranking] = sort(importance, 'descend');
    selectedBands = sort(ranking(1:numBands));
    selectedBands = selectedBands(:)'; % row vector

    fprintf('--- PCA Band Selection (Training Data Only) ---\n');
    fprintf('Input spectra: %d samples across %d spectral bands.\n', N, B);
    fprintf('PCs explaining >= %.1f%% variance: %d PCs (actual %.2f%%).\n', ...
        varianceThreshold, numPCs, cumExplained(numPCs));
    fprintf('Selected 10 common spectral bands (1-based): %s\n', mat2str(selectedBands));
end

function [coeff, explained] = native_pca(X_centered, N, B)
% Native SVD/eigendecomposition equivalent to MATLAB pca(...,'Economy',true)
    covMat = (X_centered' * X_centered) ./ (N - 1);
    [V, D] = eig(covMat);
    eigvals = diag(D);

    % Sort descending
    [eigvalsSorted, sortIdx] = sort(eigvals, 'descend');
    coeff = V(:, sortIdx);

    totalVar = sum(eigvalsSorted);
    if totalVar <= 0
        error('Zero total variance in training spectra.');
    end
    explained = 100.0 .* (eigvalsSorted ./ totalVar);
end
