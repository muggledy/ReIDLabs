clc
clear all
addpath('./mWHOS/');
load('./mWHOS/K_no_iso_gaus.mat');

imgDir = '../../../images/prid2011/single_shot/';
camADir=strcat(imgDir,'cam_a/');
camBDir=strcat(imgDir,'cam_b/');
listA = dir([camADir, '*.png']);
listA = listA(1:200);
listB = dir([camBDir, '*.png']);
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
cam_a=feats(:,1:200);
cam_b=feats(:,201:end);
save('../../../data/mwhos_prid.mat','cam_a','cam_b');