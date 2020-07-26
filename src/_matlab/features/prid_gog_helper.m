clc
clear all
addpath('./GOG/');
addpath('./GOG/mex/');

dataset_dir='../../../images/prid2011/single_shot/';
cam_a_dir=strcat(dataset_dir,'cam_a/');
cam_b_dir=strcat(dataset_dir,'cam_b/');
cam_dirs={cam_a_dir cam_b_dir};

save_dir='../../../data/gog_prid.mat';

dim=0;
params=cell(1,4);
for k=1:4
	param=set_default_parameter(k);
	params(k)={param};
    dim=dim+param.dimension;
end

cam_a=zeros(dim,200);
cam_b=zeros(dim,749);

tic
for i=1:2
    if i==1
        disp('Extract from cam_a/');
    elseif i==2
        disp('Extract from cam_b/');
    end
    start=1;
    img_files=dir(cam_dirs{i});
    if i==1
        img_files=img_files(1:200);
    end
    for j=1:length(img_files)
        img_name=img_files(j).name;
        if contains(img_name,'.png')
            if mod(start,100)==0
                if i==1
                    fprintf('Processing %d/200\n',start);
                elseif i==2
                    fprintf('Processing %d/749\n',start);
                end
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