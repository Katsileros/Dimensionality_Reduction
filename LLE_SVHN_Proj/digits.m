 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

function Y = digits(X, testX, K ,d , fid, N_train, batch_size, folder)

% Digit Dataset
fprintf(fid,'Running MNIST digits exp with K=%d, d=%d, and batch_size=%d \n',K, d, batch_size);

Y = lle([X(:,1:batch_size)], K, d, fid, folder);

end
