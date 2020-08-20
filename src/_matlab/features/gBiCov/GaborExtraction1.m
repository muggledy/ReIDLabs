function GaborFeature = GaborExtraction1(Img,ImgSize,gaborK,KernelWidth,KernelHeight)
% Extracting the Gabor features for an input image.
% Input��
%       Img: the image data�� a vector with size of [hei��wid��1] 
%       ImgSize: the size of image��width = ImgSize(1);height = ImgSize(2);
%       gaborK: the kernel of Gabor filters
% Output:
%       GaborFeature��the Gabor features��a vector with size of
%       [hei��wid��numScale��numOrientation��1]
%Notes:
%       the order of the Gabor features is orientations first, followed scales. 

% * current version��1.0
% * Author��Bingpeng MA
% * Date��2009-12-21

width = ImgSize(1);
height = ImgSize(2);
heiwid = height * width;
numGabor = size(gaborK,2);
Img = reshape(Img, height, width);
GaborFeature = zeros(heiwid*numGabor,1);
for k = 1 : numGabor
    idx = heiwid*(k-1);
    eachK = reshape(gaborK(:,k), KernelHeight, KernelWidth);
    GaborFeature_temp = gaborwavelets(Img, eachK);
    GaborFeature(idx+1 : idx+heiwid, 1) = reshape(GaborFeature_temp,heiwid,1);
end
return;