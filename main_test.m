clc; clear;
currentFolder = pwd;
addpath(genpath(currentFolder));   

%% Options before running
% Number of repeated iterations
rep_iter = 10;

%% Load existing .mat data
% S1 S2 S3 S1-N S1-L S1-T S1-UN S2-N S2-L S2-T S2-UN S3-N S3-L S3-T S3-UN
% tonedata crab wine Dermatology ionosphere seeds vehicle BUPA statlog pinus

load S1.mat

% Read: data, labels
data = di(:,1:(end-1));
label = di(:,end); 

% Initialize zero matrices for acceleration
acc = zeros(1, rep_iter);
nmi = zeros(1, rep_iter);
ARI = zeros(1, rep_iter);
Purity = zeros(1, rep_iter);

for i = 1:rep_iter

    %% Sample normalization; pre-processing, normalize the data
    data = mapminmax(data',0,1)';
    [N, dim] = size(data);
    
    % Randomly shuffle the data order
    rnd = randperm(N);
    data = data(rnd, :);
    label = label(rnd, :);
    
    % Class labels must start from 1
    if isempty(find(label < 0))
        K = length(unique(label));
    else
        % If outliers exist (label = -1)
        K = length(unique(label)) - 1;
    end

    %% Kmeans initialization of labels and normal vectors
    [U_ini, Beta_ini, initial_class] = kmeans_initial(data, K, N);
    [~, lable_initial] = max(U_ini, [], 2);

    %% RFLkPC: Robust fuzzy local K-plane clustering
    %%****Important Parameters****%%
    lambda = 10^(-2);    % Smaller lambda leads to fewer spherical clusters and more hyperplane clusters
    alpha = 0.95;
    m = 2;   % fuzzy factor
    
    t10 = tic();
    out_RFLkPC = RFlKPC(data, U_ini, K, m, alpha, lambda);
    t10 = toc(t10);
    
    [~, Iu12] = max(out_RFLkPC.U');
    [nmi(i), acc(i), ARI(i), Purity(i)] = measure(Iu12, label);
    
    RFlKPC_opt.lambda = lambda;
    RFlKPC_opt.alpha = alpha;
    ModelParOpt.RFlKPC_opt = RFlKPC_opt;

end

%% Display results
fprintf('\n========== RFLkPC Results ==========\n');
fprintf('ACC = %.4f ¡À %.4f\n', mean(acc), std(acc));
fprintf('NMI = %.4f ¡À %.4f\n', mean(nmi), std(nmi));
fprintf('ARI = %.4f ¡À %.4f\n', mean(ARI), std(ARI));
fprintf('Purity = %.4f ¡À %.4f\n', mean(Purity), std(Purity));
fprintf('Time = %.4f seconds\n', t10);

%% Visualize clustering results
figure;
gscatter(data(:,1), data(:,2), Iu12);
lgd = legend('cluster1', 'cluster2', 'cluster3');
title(lgd, 'Dataset: S1');