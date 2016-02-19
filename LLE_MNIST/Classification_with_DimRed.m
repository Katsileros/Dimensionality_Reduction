% 
%  \brief     Classification on reduced dimensionality 
%             MNIST dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      24/12/2015
%  \copyright 
%

% Executing nearest neighbor search for classification results
function err = Classification_with_DimRed(Y, s2, s2Test, N_train, N_test, batch_size, fid)

all_spaces_classification_labeling = zeros(floor(N_train ./ batch_size),N_test);

% Iter to all sub-spaces
for iter=1:floor(N_train ./ batch_size)
    
    % Nearest neighbors search
    numNeighbors = 2;

%    [IDX,~] = knnsearch(Y(:,(iter-1)*(batch_size+N_test) + 1:(iter-1)*(batch_size+N_test) + ... 
%     batch_size)',Y(:,((iter-1)*(batch_size+N_test)) + batch_size + 1:(iter*(batch_size ...
%     +N_test)))','K',numNeighbors);

    IDX = gpu_knn(Y(:,(iter-1)*(batch_size+N_test) + 1:(iter-1)*(batch_size+N_test) + batch_size),Y(:,((iter-1)*(batch_size+N_test)) + batch_size + 1:(iter*(batch_size+N_test))),numNeighbors,fid);

    IDX = IDX' + (iter-1)*batch_size;

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
    
    all_spaces_classification_labeling(iter,:) = classification_labeling(1,:); 

    for i=1:10
        error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
%         fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
    end

    fprintf(fid,'\nMean average error (subSpace-%d): %f \n', iter, (sum(error_labels) ./ 10 ) .* 100);

end

digit_labels = [1:10];
subSpace_classification_labeling = zeros(1,N_test);
error_labels = zeros(1,10);
% Subspaces voting classification
for i=1:N_test
    subSpace_voting = zeros(1,10);
    for iter=1:floor(N_train ./ batch_size)
        for j=1:10
            if(all_spaces_classification_labeling(iter,i) == digit_labels(1,j))
                subSpace_voting(1,j) = subSpace_voting(1,j) + 1;
            end
        end
    end
    
    [~,id] = max(subSpace_voting);
    subSpace_classification_labeling(1,i) = id;
    
    % Count accurate results
    if(subSpace_classification_labeling(1,i) ~= s2Test(i,1))
        error_labels(1,s2Test(i,1)) = error_labels(1,s2Test(i,1)) + 1;
    end
end

for i=1:10
    error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
%     fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
end

err =  (sum(error_labels) ./ 10 ) .* 100; 
fprintf(fid,'\nMean average error after subSpace voting is: %f \n',err);

end

