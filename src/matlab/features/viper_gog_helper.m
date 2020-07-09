% Help extract GOG features(d*n) of VIPeR(two cameras) from
% ../datasets/VIPeR.v1.0/ into ./extracted/viper_gog.mat
% muggledy 2020/6/5
clc
clear all
addpath('./GOG/');
addpath('./GOG/mex/');

dataset_dir='../../../images/VIPeR.v1.0/';
cam_a_dir=strcat(dataset_dir,'cam_a/');
cam_b_dir=strcat(dataset_dir,'cam_b/');
cam_dirs={cam_a_dir cam_b_dir};

save_dir='../../../data/gog_viper.mat';

dim=0;
params=cell(1,4);
for k=1:4
	param=set_default_parameter(k);
	params(k)={param};
    dim=dim+param.dimension;
end

cam_a=zeros(dim,632);
cam_b=zeros(dim,632);

tic
for i=1:2
    if i==1
        disp('extract from cam_a/');
    elseif i==2
        disp('extract from cam_b/');
    end
    start=1;
    img_files=dir(cam_dirs{i});
    for j=1:length(img_files)
        img_name=img_files(j).name;
        if contains(img_name,'.bmp')
            if mod(start,100)==0
                fprintf('processing %d/632\n',start);
            end
            img_name=strcat(cam_dirs{i},img_name);
            ind=1;
            for k=1:4
                t=GOG(imread(img_name),params{k});
                t_dim=params{k}.dimension;
                if i==1
                    cam_a(ind:ind+t_dim-1,start)=t;
                elseif i==2
                    cam_b(ind:ind+t_dim-1,start)=t;
                end
                ind=ind+t_dim;
            end
            start=start+1;
            %break
        end
    end
    %break
end
toc
save(save_dir,'cam_a','cam_b');