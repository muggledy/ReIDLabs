clear; clc;

imgDir = '../../../images/prid2011/single_shot/';
camADir=strcat(imgDir,'cam_a/');
camBDir=strcat(imgDir,'cam_b/');
addpath('./LOMO/');
addpath('./LOMO/bin/');

%% Get image list
listA = dir([camADir, '*.png']);
listA = listA(1:200);
listB = dir([camBDir, '*.png']);
list = [listA;listB];
n = length(list);

%% Allocate memory
info = imfinfo([camADir, list(1).name]);
images = zeros(info.Height, info.Width, 3, n, 'uint8');

%% read images
for i = 1 : n
    images(:,:,:,i) = imread([list(i).folder, '\', list(i).name]);
end

%% extract features. Run with a set of images is usually faster than that one by one, but requires more memory.
descriptors = LOMO(images);

%% if you need to set different parameters other than the defaults, set them accordingly
%{
options.numScales = 3;
options.blockSize = 10;
options.blockStep = 5;
options.hsvBins = [8,8,8];
options.tau = 0.3;
options.R = [3, 5];
options.numPoints = 4;

descriptors = LOMO(images, options);
%}
cam_a=descriptors(:,1:200);
cam_b=descriptors(:,201:end);
save('../../../data/lomo_prid.mat','cam_a','cam_b');