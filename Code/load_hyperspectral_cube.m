function [cube, info] = load_hyperspectral_cube(bilPath, hdrPath)
%LOAD_HYPERSPECTRAL_CUBE Loads a BIL hyperspectral image into an H x W x 168 array.
%
%   [cube, info] = load_hyperspectral_cube(bilPath)
%   [cube, info] = load_hyperspectral_cube(bilPath, hdrPath)
%
% INPUTS:
%   bilPath - Path to the binary .bil hyperspectral file.
%   hdrPath - (Optional) Path to the associated ENVI .hdr header file.
%             If omitted, the function replaces '.bil' with '.hdr'.
%
% OUTPUTS:
%   cube - H x W x 168 double array containing spectral measurements.
%   info - Struct containing metadata parsed from the header (lines, samples, bands, etc.)
%
% This function supports both MATLAB's Hyperspectral Imaging Library (enviinfo/imhypercube)
% and a standalone native binary reader for maximum portability across any MATLAB environment.

    if nargin < 2 || isempty(hdrPath)
        [fdir, fname, ~] = fileparts(bilPath);
        hdrPath = fullfile(fdir, [fname, '.hdr']);
    end

    if ~exist(bilPath, 'file')
        error('BIL file not found: %s', bilPath);
    end
    if ~exist(hdrPath, 'file')
        error('HDR file not found: %s', hdrPath);
    end

    % Parse ENVI HDR header metadata
    info = parse_envi_hdr(hdrPath);

    H = info.lines;
    W = info.samples;
    B = info.bands;

    % Method 1: Use MATLAB Support Package if available
    usedBuiltin = false;
    if exist('imhypercube', 'file') && exist('enviinfo', 'file')
        try
            hcube = imhypercube(bilPath);
            cube = double(gather(hcube));
            usedBuiltin = true;
        catch
            usedBuiltin = false;
        end
    end

    % Method 2: Native robust BIL binary reader
    if ~usedBuiltin
        fid = fopen(bilPath, 'r', 'ieee-le');
        if fid == -1
            error('Failed to open BIL binary file: %s', bilPath);
        end
        % BIL format: Band Interleaved by Line
        % Layout on disk: for line 1..H: for band 1..B: sample 1..W
        % Total elements = H * B * W of uint16
        raw = fread(fid, [W * B, H], 'uint16=>double');
        fclose(fid);

        % Reshape to [W, B, H] then permute to [H, W, B]
        raw = reshape(raw, [W, B, H]);
        cube = permute(raw, [3, 1, 2]);
    end

    % Validate output dimensions
    [outH, outW, outB] = size(cube);
    if outH ~= H || outW ~= W || outB ~= B
        error('Dimension mismatch: expected %dx%dx%d, got %dx%dx%d', H, W, B, outH, outW, outB);
    end
end

function info = parse_envi_hdr(hdrPath)
% Helper to parse ENVI header file text into a struct
    fid = fopen(hdrPath, 'r');
    if fid == -1
        error('Cannot open HDR file: %s', hdrPath);
    end
    rawText = fread(fid, '*char')';
    fclose(fid);

    info = struct();
    info.lines = 0;
    info.samples = 0;
    info.bands = 168;
    info.byteOrder = 0;
    info.dataType = 12; % uint16
    info.interleave = 'bil';

    lines = strsplit(rawText, {'\n', '\r'});
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if contains(line, '=')
            parts = strsplit(line, '=');
            key = lower(strtrim(parts{1}));
            val = strtrim(parts{2});
            switch key
                case 'lines'
                    info.lines = str2double(val);
                case 'samples'
                    info.samples = str2double(val);
                case 'bands'
                    info.bands = str2double(val);
                case 'byte order'
                    info.byteOrder = str2double(val);
                case 'data type'
                    info.dataType = str2double(val);
                case 'interleave'
                    info.interleave = lower(val);
                case 'description'
                    info.description = val;
            end
        end
    end
end
