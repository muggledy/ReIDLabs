function Gaborout  = gaborwavelets(img, G)
% Get the Gabor features using the Gabor kernel
% Input:
%       img: the image, a matrix with size of [width, height]
%       G: the kernel of Gabor
% Output:
%       Gaborout: the Gabor features

% Reference:
%       Laurenz Wiskott, Jean-Marc Fellous,Norbert Kruger, and Christoph von der Malsburg,
%               "Face Recognition by Elastic Bunch Graph Matching"
% * current version��1.0
% * Author��Bingpeng MA
% * Date��2009-12-21
if isa(img,'double')~=1
    img = im2double(img);
end
Imgabout = conv2(img,double(imag(G)), 'same');
Regabout = conv2(img,double(real(G)), 'same');
Gaborout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);
return;