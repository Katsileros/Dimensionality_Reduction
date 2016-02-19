%
%  \brief     Classification on reduced dimensionality
%             Arcene dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      03/02/2016
%  \copyright
%

% Executing nearest neighbor search for classification results
function acc = Classification_with_DimRed(Y, s2, s2Test, N_train, N_test, fid)

% Nearest neighbors search
numNeighbors = 4;

[IDX,~] = knnsearch(Y(:,1:N_train)',Y(:,N_train+1:N_train+N_test)','K',numNeighbors);
% IDX = gpu_knn(Y(:,1:N_train), Y(:,N_train+1:N_train+N_test), numNeighbors,fid);
% IDX = IDX';

% Arcene labels
labels = [-1,1];
classification_labeling = zeros(1,N_test);

for i=1:N_test

    nn_votings = zeros(1,length(labels));
    for k=1:numNeighbors
	for j=1:length(labels)
	  if(s2(IDX(i,k),1) == labels(1,j))
		nn_votings(1,j) = nn_votings(1,j) + 1;
	  end
	end
    end
    
    [~,classification_labeling(1,i)] = max(nn_votings);
    
end
classification_labeling(1,classification_labeling == 1) = -1;
classification_labeling(1,classification_labeling == 2) = 1;

err = sum((classification_labeling == s2Test') == 1);
acc = (err ./ N_test) * 100;
fprintf(fid,'\n Accuracy (per cent) with dimensionality reduction: %f \n', acc);
% fprintf('\n Accuracy with dimensionality reduction: %f \n', (err ./ N_test) * 100);

end

