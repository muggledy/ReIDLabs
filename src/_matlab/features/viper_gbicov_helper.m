% Help extract gBiCov features(d*n) of VIPeR(two cameras) from
% .../VIPeR.v1.0/ into .../gbicov_viper.mat
% 结果非常低，rank-1只有10.7%。。
% muggledy 2020/8/5
clc
clear all
addpath('./gBiCov/');

imgDir = '../../../images/VIPeR.v1.0/';
camADir=strcat(imgDir,'cam_a/');
camBDir=strcat(imgDir,'cam_b/');
listA = dir([camADir, '*.bmp']);
listB = dir([camBDir, '*.bmp']);
list = [listA;listB];
n = length(list);
info = imfinfo([camADir, list(1).name]);
images = cell(n,1);

for i = 1 : n
    images{i} = imread([list(i).folder, '\', list(i).name]);
end

feats=gbicovEx(images,[20,20],[10,10]);

cam_a=feats(:,1:632);
cam_b=feats(:,633:end);
save('../../../data/gbicov_viper.mat','cam_a','cam_b');