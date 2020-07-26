clc
clear all
addpath('./mWHOS/');
load('./mWHOS/K_no_iso_gaus.mat');

imgDir = '../../../images/VIPeR.v1.0/';
camADir=strcat(imgDir,'cam_a/');
camBDir=strcat(imgDir,'cam_b/');
listA = dir([camADir, '*.bmp']);
listB = dir([camBDir, '*.bmp']);
list = [listA;listB];
n = length(list);
info = imfinfo([camADir, list(1).name]);

feats=[];
for i = 1 : n
    img = imread([list(i).folder, '\', list(i).name]);
    if mod(i,20)==0
        fprintf('extract img %d/%d\n',i,n);
    end
    feats=[feats;PETA_cal_img_full_hist(img,K_no_iso_gaus,1)];
end

feats=feats';
cam_a=feats(:,1:632);
cam_b=feats(:,633:end);
save('../../../data/mwhos_viper.mat','cam_a','cam_b');