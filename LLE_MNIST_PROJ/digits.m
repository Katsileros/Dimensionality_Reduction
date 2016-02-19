 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

function Y = digits(X, K ,d , fid, N_train, batch_size, folder)

% Digit Dataset
fprintf(fid,'Running MNIST digits exp with K=%d, d=%d, and batch_size=%d \n',K, d, batch_size);

%tmpX = zeros(floor(N_train ./ batch_size), size(X,1), size(X,2));
% Preallocation for parallel for
%for i=1:floor(N_train ./ batch_size)
%    tmpX(i,:,(i-1)*batch_size + 1 : i*batch_size) = X(:,(i-1)*batch_size + 1 : i*batch_size);
%end

% Y = lle([X(:,1:batch_size) testX], K, d, fid, folder);
Y = lle(X(:,1:batch_size), K, d , fid, folder);
for i=2:floor(N_train ./ batch_size)
    tmp = lle(X(:,(i-1)*batch_size + 1 : i*batch_size), K, d, fid, folder);
    % tmp = lle([X(:,(i-1)*batch_size + 1 : (i*batch_size)) testX], K, d, fid, folder);
    %  tmp = lle(tmpX(i,:,(i-1)*batch_size + 1 : i*batch_size), K, d, fid, folder);
    Y = [Y tmp];
end
save(strcat(folder,'Y.mat'),'Y');

end
