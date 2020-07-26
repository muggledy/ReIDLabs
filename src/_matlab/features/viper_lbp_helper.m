clc
clear all
addpath('./LBP/');

imgDir = '../../../images/VIPeR.v1.0/';
camADir=strcat(imgDir,'cam_a/');
camBDir=strcat(imgDir,'cam_b/');
listA = dir([camADir, '*.bmp']);
listB = dir([camBDir, '*.bmp']);
list = [listA;listB];
n = length(list);
info = imfinfo([camADir, list(1).name]);

feats=[];
mapping=getmapping(8,'u2');
for i = 1 : n
    img = imread([list(i).folder, '\', list(i).name]);
    feat1=lbp(img,1,8,mapping,'h');
    feat2=lbp(rgb2hsv(img),1,8,mapping,'h');
    feat3=lbp(rgb2lab(img),1,8,mapping,'h');
%     feat4=lbp(rgb2ycbcr(img),1,8,mapping,'h');
    feats=[feats;feat1,feat2,feat3];%,feat4];
end

feats=feats';
cam_a=feats(:,1:632);
cam_b=feats(:,633:end);
save('../../../data/lbp_viper.mat','cam_a','cam_b');