Elyor, 01/05/2017
I implemented the feature extraction code for the paper [1]. 
The fearures may not be perfect implementation. 
The features are used for our papers [2,3]. 
If you use the code please cite the papers. 

===============
1) To run this code:
+ Include vl_feat library : http://www.vlfeat.org/download.html. 
   Firstly, download and unpack the latest VLFeat binary distribution into a directory of your choice (e.g. ~/src/vlfeat). Let VLFEATROOT denote this directory.
   Then edit startup.m in userpath(this is a matlab command) with:
       run('VLFEATROOT/toolbox/vl_setup');
       fprintf(strcat('Installed VLFeat(version:',vl_version,').\n'));
   Now when you restart MATLAB everytime, you should see the VLFeat installed automatically.
+ mex HoG.cpp according to your machine (I include linux and windows version).

[1] Giuseppe Lisanti et al., Matching People across Camera Views using Kernel Canonical Correlation Analysis”, Eighth ACM/IEEE International Conference on Distributed Smart Cameras, 2014.
[2] Elyor Kodirov et al., Dictionary Learning with Iterative Laplacian Regularisation for Unsupervised Person Re-identification, BMVC 2015
[3] Elyor Kodirov et al., Person Re-identification by Unsupervised l_1 Graph Learning, ECCV2016.


