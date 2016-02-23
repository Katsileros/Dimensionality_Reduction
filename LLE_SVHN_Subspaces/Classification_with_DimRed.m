%
%  \brief     Classification on reduced dimensionality
%             SVHN dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      19/02/2016
%  \copyright


% Executing nearest neighbor search for classification results
function err = Classification_with_DimRed(train_descr, test_descr, train_labels, test_labels, Ntrain, Ntest, batch_size, dim, fid, folder)

% Make train data
tmp = train_descr;
clear train_descr;
train_descr = zeros(dim,Ntrain);

for i=1:Ntrain
    train_descr(:,i) = tmp{i,1}(1:dim,1);
end

% Make test data
tmp = test_descr;
clear test_descr;
test_descr = zeros(dim,Ntest);

for i=1:Ntest
    test_descr(1:dim,i) = tmp{i,1}(1:dim,1);
end
clear tmp;

all_spaces_classification_labeling = zeros(floor(Ntrain ./ batch_size),(Ntest./(floor(Ntrain ./ batch_size))));

% Iter to all sub-spaces
for iter=1:floor(Ntrain ./ batch_size)
    N_train = Ntrain ./ (floor(Ntrain ./ batch_size));
    N_test = Ntest ./ (floor(Ntrain ./ batch_size));
    
    % Nearest neighbors search
    numNeighbors = 8;
    [IDX,~] = knnsearch(train_descr(:,(iter-1)*batch_size+1:(iter-1)*batch_size + batch_size)',test_descr(:,(iter-1)*N_test+1:iter*N_test)','K',numNeighbors);
    IDX = IDX + (iter-1)*batch_size;
    
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
    
    all_spaces_classification_labeling(iter,:) = classification_labeling(1,:); 

    for i=1:10
        error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
        fprintf(fid,'\nError for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
    end
   
    save(strcat(folder,'error_labels_',num2str(iter),'.mat'),'error_labels');
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
    
    [~,subSpace_classification_labeling(1,i)] = max(subSpace_voting);
    
    % Count accurate results
    if(subSpace_classification_labeling(1,i) ~= test_labels(i,1))
        error_labels(1,test_labels(i,1)) = error_labels(1,test_labels(i,1)) + 1;
    end
end

for i=1:10
    error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
%     fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
end

err =  (sum(error_labels) ./ 10 ) .* 100; 
fprintf(fid,'Mean average error after subSpace voting is: %f \n',err);
fprintf('Mean average error after subSpace voting is: %f \n\n',err);

end
