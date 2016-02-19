 % 
 %  \brief     Classification on MNIST dataset with NN algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

% Executing nearest neighbor search for classification results
function Classification_without_DimRed(X, testX, s2, s2Test, N_test, fid)

% Nearest neighbors search
numNeighbors = 2;

 % tic;
 % [IDX,~] = knnsearch(X',testX','K',numNeighbors);
 % toc;
 % %size(IDX)

IDX = gpu_knn(X,testX,numNeighbors,fid);
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
    
fprintf(fid,'\nMean average error (without dimensionality reduction): %f \n', (sum(error_labels) ./ 10 ) .* 100);

end
