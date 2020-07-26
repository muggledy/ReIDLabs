clc
clear all
addpath('./mWHOS/');
load('./mWHOS/K_no_iso_gaus.mat');

imgDir = '../../../images/CUHK01/';
list = dir([imgDir, '*.png']);
n = length(list);
info = imfinfo([imgDir, list(1).name]);

feats=[];
for i = 1 : n
    img = imread([list(i).folder, '\', list(i).name]);
    if mod(i,20)==0
        fprintf('extract img %d/%d\n',i,n);
    end
    feats=[feats;PETA_cal_img_full_hist(img,K_no_iso_gaus,1)];
end

feats=feats';
a1=(1:4:3884);
a2=(2:4:3884);
b1=(3:4:3884);
b2=(4:4:3884);
a=reshape([a1;a2],1,1942);
b=reshape([b1;b2],1,1942);

cam_a=feats(:,a);
cam_b=feats(:,b);
save('../../../data/mwhos_cuhk01.mat','cam_a','cam_b');