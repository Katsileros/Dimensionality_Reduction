 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

function Y = digits(X, testX, K ,d , fid, folder)

% Digit Dataset
fprintf(fid,'Running MNIST digits exp with K=%d, d=%d \n',K, d);

Y = lle([X testX], K, d , fid, folder);
save(strcat(folder,'Y.mat'),'Y');

end
