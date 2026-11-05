function [paddedCube, padDetails] = pad_hyperspectral_image(cube, targetSize, padValue, normScale)
%PAD_HYPERSPECTRAL_IMAGE Standardizes spatial dimensions using balanced padding only.
%
%   [paddedCube, padDetails] = pad_hyperspectral_image(cube, targetSize)
%   [paddedCube, padDetails] = pad_hyperspectral_image(cube, targetSize, padValue, normScale)
%
% INPUTS:
%   cube        - H x W x C numeric array (e.g. H x W x 10 hyperspectral bands).
%   targetSize  - [HT, WT] target spatial dimensions (default: [74, 74]).
%   padValue    - Constant fill value for padding border (default: 0.0).
%   normScale   - Normalization divisor (default: 65535.0 to scale raw uint16 to [0,1]).
%                 Set to 1.0 or empty if normalization is handled separately.
%
% OUTPUTS:
%   paddedCube  - HT x WT x C double array. Original data is centered with balanced
%                 padding on opposing edges. No resizing, interpolation, or cropping.
%   padDetails  - Struct containing padding offsets: padTop, padBottom, padLeft, padRight.
%
% INFERENCE DETECTION:
%   Detects and reports an error if the input spatial dimensions exceed targetSize.

    if nargin < 2 || isempty(targetSize)
        targetSize = [74, 74];
    end
    if nargin < 3 || isempty(padValue)
        padValue = 0.0;
    end
    if nargin < 4 || isempty(normScale)
        normScale = 65535.0;
    end

    [H, W, C] = size(cube);
    HT = targetSize(1);
    WT = targetSize(2);

    % Inference Check: detect if input exceeds established target dimensions
    if H > HT || W > WT
        error('Input image dimensions [%d, %d] exceed target dimensions [%d, %d]. Cropping is not permitted.', ...
            H, W, HT, WT);
    end

    % Normalize original valid pixels (retain spectral relationships without distortion)
    if normScale > 1.0
        normCube = double(cube) ./ normScale;
    else
        normCube = double(cube);
    end

    % Calculate balanced padding across opposing edges
    totalPadH = HT - H;
    totalPadW = WT - W;

    padTop = floor(totalPadH / 2);
    padBottom = totalPadH - padTop;

    padLeft = floor(totalPadW / 2);
    padRight = totalPadW - padLeft;

    % Allocate output array pre-filled with padValue
    paddedCube = ones(HT, WT, C, 'like', normCube) * padValue;

    % Insert original uncropped, un-interpolated data into the balanced center
    rowStart = padTop + 1;
    rowEnd   = padTop + H;
    colStart = padLeft + 1;
    colEnd   = padLeft + W;

    paddedCube(rowStart:rowEnd, colStart:colEnd, :) = normCube;

    padDetails = struct();
    padDetails.originalSize = [H, W, C];
    padDetails.targetSize = [HT, WT, C];
    padDetails.padTop = padTop;
    padDetails.padBottom = padBottom;
    padDetails.padLeft = padLeft;
    padDetails.padRight = padRight;
    padDetails.padValue = padValue;
    padDetails.normScale = normScale;
end
