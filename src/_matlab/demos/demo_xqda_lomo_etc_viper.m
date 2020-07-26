% This is a demo for the XQDA metric learning, as well as the evaluation on 
% the VIPeR database with LOMO,GOG etc. descriptors. Firstly you have to run
% ../features/viper_lomo_helper.m,../features/viper_gog_helper.m etc. to get
% viper's LOMO,GOG etc. features.
% Note: cam_b as probe here

close all; clear; clc;

dataset='mwhos';

addpath('../metric_learning/');
addpath('../tools/');
feaFile = strcat('../../../data/',dataset,'_viper.mat');

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted features
load(feaFile);
galFea = cam_a';
probFea = cam_b';
clear cam_a
clear cam_b

%% set the seed of the random stream
% seed = 0;
% rng(seed);

%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
    p = randperm(numClass);
    
    galFea1 = galFea( p(1:numClass/2), : );
    probFea1 = probFea( p(1:numClass/2), : );
    
    t0 = tic;
    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');
    
    clear galFea1 probFea1
    trainTime = toc(t0);
    
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probFea2 = probFea(p(numClass/2+1 : end), : );
    
    t0 = tic;
    dist = MahDist(M, galFea2 * W, probFea2 * W);
    
    clear galFea2 probFea2 M W
    matchTime = toc(t0);
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    
    cms(nf,:) = EvalCMC( -dist, 1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
