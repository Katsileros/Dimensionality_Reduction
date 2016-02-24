 % 
 %  \brief     Classification on SVHN dataset with NN algorithm
 %  \author    Katsileros Petros
 %  \date      13/02/2016
 %  \copyright 
 %

% Executing nearest neighbor search for classification results
function err = Classification_without_DimRed(train_descr, test_descr, train_labels, test_labels, N_train, N_test, fid)

% Make train data
tmp = train_descr;
clear train_descr;
train_descr = zeros(size(tmp{1,1},2),N_train);

for i=1:N_train
    train_descr(:,i) = tmp{i,1};
end

% Make test data
tmp = test_descr;
clear test_descr;
test_descr = zeros(size(tmp{1,1},2),N_test);

for i=1:N_test
    test_descr(:,i) = tmp{i,1};
end
clear tmp;

% Nearest neighbors search
numNeighbors = 8;
[IDX,~] = knnsearch(train_descr',test_descr','K',numNeighbors);
% IDX = gpu_knn(train_descr,test_descr,numNeighbors,fid);
% IDX = IDX(1:numNeighbors,:)';

% SVHN labels
digit_labels = [1:10];
classification_labeling = zeros(1,N_test);
error_labels = zeros(1,10);
overall_digit_labels = zeros(1,10);

for i=1:N_test

    nn_votings = zeros(1,10);
    for k=1:numNeighbors
	for j=1:10
	  if(train_labels(IDX(i,k),1) == digit_labels(1,j))
		nn_votings(1,j) = nn_votings(1,j) + 1;
	  end
	end
    end
    
    [~,classification_labeling(1,i)] = max(nn_votings);
    
    overall_digit_labels(1,test_labels(i,1)) = overall_digit_labels(1,test_labels(i,1)) + 1;

    % Count accurate results
    if(classification_labeling(1,i) ~= test_labels(i,1))
        error_labels(1,test_labels(i,1)) = error_labels(1,test_labels(i,1)) + 1;
    end
end

for i=1:10
    error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
    % fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
end

err = (sum(error_labels) ./ 10 ) .* 100;
fprintf(fid,'\nMean average error without dimensionality reduction: %f \n', err);
fprintf('Mean average error without dimensionality reduction: %f \n\n', err);

end
