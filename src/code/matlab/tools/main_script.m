load('../../../../data/XS.mat');
D = learn_basis(X,S,1);
save('../../../../data/XS_D.mat','D');