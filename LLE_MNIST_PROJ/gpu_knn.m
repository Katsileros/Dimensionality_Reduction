% 
%  \brief     GPU knn, using knn-toolbox
%  \author    Katsileros Petros
%  \date      5/11/2015
%  \copyright 
%

function idxHS = gpu_knn(X,Y,k,fid)

fprintf(fid,'Running knn on GPU (K=%d) \n',k);

streams = 1;

% [~,idxHS,~] = gpuknnHeap(Y, X, k, streams);
[~,idxHS,~] = gpuknnBitonic(Y, X, k, streams);

end
