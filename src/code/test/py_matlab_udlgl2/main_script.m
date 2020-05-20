load('../../../../data/graphsc_para.mat');
[D, ~, ~] = GraphSC_cor(X_tr, W_full, nBasis, alpha, beta, nIters);
save('../../../../data/graphsc_result.mat','D');