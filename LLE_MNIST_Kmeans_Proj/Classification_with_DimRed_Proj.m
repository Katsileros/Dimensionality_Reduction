%
%  \brief     Classification on reduced dimensionality
%             MNIST dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      05/01/2016
%  \copyright
%

% Executing nearest neighbor search for classification results
function err = Classification_with_DimRed_Proj(Y, X, testX, s2, s2Test, N_train, N_test, fid)

% Nearest neighbors search
numNeighbors = 4;

% [IDX,~] = knnsearch(X',testX','K',numNeighbors);

IDX = gpu_knn(X, testX, numNeighbors,fid);
IDX = IDX';

%% Project test data
nn_graph = zeros(N_train,size(testX,2));
testY = zeros(size(Y,1),size(testX,2));
% Loop for every test data
for i=1:size(testX,2)
    nn_graph(IDX(i,:),i) = 1;
    testY(:,i) = Y * nn_graph(:,i);
end

% Nearest neighbors search
numNeighbors = 4; 
% [IDX1,~] = knnsearch(Y',testY','K',numNeighbors);

IDX = gpu_knn(Y, testY, numNeighbors,fid);
IDX = IDX';

% MNIST labels
digit_labels = [1:10];
classification_labeling = zeros(1,N_test);
error_labels = zeros(1,10);
overall_digit_labels = zeros(1,10);

for i=1:N_test
    % Digit votings
    nn_votings = zeros(1,10);
    for k=1:numNeighbors
        for j=1:10
            if(s2(IDX(i,k),1) == digit_labels(1,j))
                nn_votings(1,j) = nn_votings(1,j) + 1;
            end
        end
    end
    
    [~,id] = max(nn_votings);
    classification_labeling(1,i) = id;
    
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
fprintf(fid,'\nMean average error with dimensionality reduction, kmeans clustering and projections: %f \n', err);


end

