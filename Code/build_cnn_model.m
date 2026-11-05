function lgraph = build_cnn_model(architectureType, inputSize, numClasses, dropoutRate)
%BUILD_CNN_MODEL Creates a 2D CNN architecture for polymer classification.
%
%   lgraph = build_cnn_model()
%   lgraph = build_cnn_model(architectureType, inputSize, numClasses, dropoutRate)
%
% INPUTS:
%   architectureType - 'shallow', 'medium' (default baseline), or 'deep'.
%   inputSize        - [HT, WT, C] input tensor size (default: [74, 74, 10]).
%   numClasses       - Number of target classes (default: 4).
%   dropoutRate      - Dropout probability before final FC layer (default: 0.3).
%
% OUTPUT:
%   lgraph           - MATLAB layerGraph or Layer array ready for training.

    if nargin < 1 || isempty(architectureType)
        architectureType = 'medium';
    end
    if nargin < 2 || isempty(inputSize)
        inputSize = [74, 74, 10];
    end
    if nargin < 3 || isempty(numClasses)
        numClasses = 4;
    end
    if nargin < 4 || isempty(dropoutRate)
        dropoutRate = 0.3;
    end

    if ~exist('imageInputLayer', 'file')
        warning('MATLAB Deep Learning Toolbox imageInputLayer not found. Returning structural layer specification.');
        lgraph = struct('type', architectureType, 'inputSize', inputSize, ...
                        'numClasses', numClasses, 'dropoutRate', dropoutRate);
        return;
    end

    layers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    ];

    switch lower(architectureType)
        case 'shallow'
            % 2 Convolutional Blocks (16 -> 32)
            layers = [
                layers
                % Block 1
                convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

                % Block 2
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutRate, 'Name', 'dropout')
                fullyConnectedLayer(numClasses, 'Name', 'fc')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];

        case 'medium'
            % 3 Convolutional Blocks (16 -> 32 -> 64) - Baseline Model
            layers = [
                layers
                % Block 1
                convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

                % Block 2
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

                % Block 3
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutRate, 'Name', 'dropout')
                fullyConnectedLayer(numClasses, 'Name', 'fc')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];

        case 'deep'
            % 4 Convolutional Blocks (16 -> 32 -> 64 -> 128)
            layers = [
                layers
                % Block 1
                convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

                % Block 2
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

                % Block 3
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

                % Block 4
                convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
                batchNormalizationLayer('Name', 'bn4')
                reluLayer('Name', 'relu4')
                maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

                globalAveragePooling2dLayer('Name', 'gap')
                dropoutLayer(dropoutRate, 'Name', 'dropout')
                fullyConnectedLayer(numClasses, 'Name', 'fc')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];

        otherwise
            error('Unknown architectureType: %s. Use shallow, medium, or deep.', architectureType);
    end

    lgraph = layerGraph(layers);
end
