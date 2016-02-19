% 
%  \brief     GPU knn, using knn-toolbox
%  \author    Katsileros Petros
%  \date      5/11/2015
%  \copyright 
%

function idxHS = gpu_knn(X,Y,k,fid)

fprintf(fid,'Running knn on GPU (K=%d) \n',k);

streams = 1;

if(k <= 16)
	nn = 16;
elseif(k <= 32)
	nn = 32;
elseif(k <= 64)
	nn = 64;
elseif(k <= 128)
	nn = 128;
elseif(k <= 256)
	nn = 256;
elseif(k <= 512)
	nn = 512;
else
	fprintf('Nearest neighbors must be <= 512. \n');
	exit(0);
end

[~,idxHS,~] = gpuknnHeap(Y, X, nn, streams);
%[~,idxHS,~] = gpuknnBitonic(Y, X, nn, streams);

end
