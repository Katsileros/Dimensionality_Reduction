 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

function Y = digits(X, K ,d , fid)

% Digit Dataset
fprintf(fid,'Running MNIST digits exp with K=%d, d=%d \n',K, d);

Y = lle(X, K, d, fid);
% noDim red test
% Y = X;

end
