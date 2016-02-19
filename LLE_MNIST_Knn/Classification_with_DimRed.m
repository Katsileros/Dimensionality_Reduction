%
%  \brief     Classification on reduced dimensionality
%             MNIST dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      05/01/2016
%  \copyright
%

% Executing nearest neighbor search for classification results
function err = Classification_with_DimRed(Y, s2, s2Test, N_train, N_test, fid)

% Nearest neighbors search
numNeighbors = 64;

% [IDX1,~] = knnsearch(Y(:,1:N_train)',Y(:,N_train+1:N_train+N_test)','K',numNeighbors);
IDX = gpu_knn(Y(:,1:N_train), Y(:,N_train+1:N_train+N_test), numNeighbors,fid);
IDX = IDX';

% MNIST labels
digit_labels = [1:10];
classification_labeling = zeros(1,N_test);
error_labels = zeros(1,10);
overall_digit_labels = zeros(1,10);

for i=1:N_test
    % Nearest cluster members
%    nn_votings = zeros(1,10);
%    clust_labels = s2(assign == assign(IDX(i,1)));
%    nn_votings = histc(clust_labels',[1:10]);

    nn_votings = zeros(1,10);
    for k=1:numNeighbors
	for j=1:10
	  if(s2(IDX(i,k),1) == digit_labels(1,j))
		nn_votings(1,j) = nn_votings(1,j) + 1;
	  end
	end
    end
    
    [~,classification_labeling(1,i)] = max(nn_votings);
    
    overall_digit_labels(1,s2Test(i,1)) = overall_digit_labels(1,s2Test(i,1)) + 1;

    % Count accurate results
    if(classification_labeling(1,i) ~= s2Test(i,1))
        error_labels(1,s2Test(i,1)) = error_labels(1,s2Test(i,1)) + 1;
    end
end

for i=1:10
    error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
    % fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
end

err = (sum(error_labels) ./ 10 ) .* 100;
fprintf(fid,'\nMean average error with dimensionality reduction and kmeans clustering: %f \n', err);
% fprintf('Mean average error with dimensionality reduction and kmeans clustering: %f \n', err);

end

