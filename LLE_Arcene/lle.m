% 
%  \brief     Locally Linear Embedding algorithm
%  \source    https://www.cs.nyu.edu/~roweis/lle/algorithm.html
%  \author    Katsileros Petros
%  \date      14/6/2015
%  \copyright 
%

% LLE ALGORITHM (using K nearest neighbors)
%
% [Y] = lle(X,K,dmax)
%
% X = data as D x N matrix (D = dimensionality, N = #points)
% K = number of neighbors
% dmax = max embedding dimensionality
% Y = embedding as d x N matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Y] = lle(X,K,d,fid,folder)

[D,N] = size(X);
fprintf(fid,'LLE running on %d points with %d dimensions\n',N,D);

% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
fprintf(fid,'-->Finding %d nearest neighbours.\n',K);

% Serial knn
tic;
[IDX,~] = knnsearch(X',X','K',K+1);
neighborhood = IDX(:,2:end)';
elapsedTime = toc;
fprintf(fid,'Elapsed time is: %6f \n',elapsedTime);

% GPU knn
% tic
% if(K<= 16)
%   idxHS = gpu_knn(X,X,16,fid);
% elseif(K>16)
%   idxHS = gpu_knn(X,X,32,fid);
% elseif(K>32)
%   idxHS = gpu_knn(X,X,64,fid);
% elseif(K>64)
%   idxHS = gpu_knn(X,X,128,fid);
% elseif(K>128)
%   idxHS = gpu_knn(X,X,256,fid);
% else
%   fprintf('K>256 (exit) \n');
%   pause;
%   exit(0);
% end

% neighborhood = idxHS(2:K,:);
% clear idxHS;
% K = K - 1;
% elapsedTime = toc;
% fprintf(fid,'Elapsed time is: %6f \n',elapsedTime);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
fprintf(fid,'-->Solving for reconstruction weights.\n');

if(K>D) 
  fprintf(fid,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
%    C = C + tol*diag(diag(C)); 
   W(:,ii) = C\ones(K,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end;

% clear X;

% STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
fprintf(fid,'-->Computing embedding.\n');

tic;
% M=eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w*w';
end;
elapsedTime = toc;
fprintf(fid,'Elapsed time is: %6f \n',elapsedTime);

% CALCULATION OF EMBEDDING
options.disp = 0; options.isreal = 1; options.issym = 1; 
fprintf(fid,'Eigen value decomposition ...\n');

tic;
[Y,eigenvals] = eigs(M,d+1,0,options);
elapsedTime = toc;
fprintf(fid,'Elapsed time is: %6f \n',elapsedTime);

% M*Y - Y*eigenvals

save(strcat(folder,'eigenvals.mat'),'eigenvals');
% eigenvals

% Y = Y(:,2:d+1)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0
Y = Y(:,end-3:-1:end-d)'*sqrt(N);

fprintf(fid,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% other possible regularizers for K>D
%   C = C + tol*diag(diag(C));                       % regularlization
%   C = C + eye(K,K)*tol*trace(C)*K;                 % regularlization
