function person_rep = PETA_cal_img_full_hist(im, K_no_iso_gaus, HOG_LBP)
    % Elyor, 07/02/2015
    % HOG_LBP = 1 means you include HOG and LBP features
    % im = image
    % K_no_iso_gaus = kernel
    
    im_rgb = (imresize(im,[128 64])); % ###based on paper
    person_rep = [];
    
    % Color
    k=1;
    for i = 1:8 % 1 level
        im_part = im_rgb(k:i*16,:,:);
        im_hist_weight = K_no_iso_gaus(k:i*16,:);
        person_rep = [person_rep PETA_cal_color_histogram(im_part, im_hist_weight)];
        k = i*16;
    end
    
    % Reduction
    im_rgb_reduced = im_rgb(9:120,9:56,:); % ###based on paper
    K_no_iso_gaus_reduced =K_no_iso_gaus(9:120,9:56,:); 
    k=1;
    for i = 1:7 % 2 level
        im_part = im_rgb_reduced(k:i*16,:,:);
        im_hist_weight = K_no_iso_gaus_reduced(k:i*16,:);
        person_rep = [person_rep PETA_cal_color_histogram(im_part, im_hist_weight)];
        k = i*16;
    end
    
    if HOG_LBP == 1
        % HOG
        HOG_Fea.Params = [4 8 2 0 0.25]; % ###based on paper
        HOG_Fea   = HoG(double(im_rgb_reduced),HOG_Fea.Params)';
        person_rep = [person_rep HOG_Fea];
        
        % LBP
        LBP_Fea.cellSize = 16;           % ###based on paper
        LBP_Fea  = double(vl_lbp(single(rgb2gray(im_rgb_reduced)), LBP_Fea.cellSize));
        person_rep = [person_rep reshape(LBP_Fea, [1 7*3*58])];
    end
        
    person_rep = sqrt(person_rep);
end