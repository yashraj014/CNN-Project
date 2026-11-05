%% RUN_EXPERIMENTS Displays controlled CNN experimental comparisons
%
% Task 6: Controlled comparisons changing one factor at a time:
%   - Comparison 1: Network Depth (Shallow vs Medium vs Deep)
%   - Comparison 2: Regularization / Dropout (0.0 vs 0.3 vs 0.6)
%   - Comparison 3: Learning Rate (1e-4 vs 1e-3 vs 1e-2)

fprintf('========================================================================\n');
fprintf('  Task 6: Controlled CNN Experimental Investigation (ME3101)\n');
fprintf('========================================================================\n\n');

resCsvPath = fullfile('Results', 'experimental_results.csv');
if ~exist(resCsvPath, 'file')
    error('Results file not found: %s', resCsvPath);
end

% Read and display results table
T = readtable(resCsvPath);
disp(T(:, {'Experiment_ID', 'Category', 'Test_Accuracy', 'Avg_Class_Acc_Mean_Recall', 'Macro_F1_Score', 'Weighted_F1_Score'}));
