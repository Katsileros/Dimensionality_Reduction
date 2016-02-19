tic
N = 10000;

load('Y.mat');
% train_data = X;
train_data = Y(:,1:N);
load('labels.mat');
labels = labels(1:N,:)';

% Create a Pattern Recognition Network
% hiddenLayerSize = [440 240 120];
% hiddenLayerSize = [240 140 80];
% hiddenLayerSize = [120 40];
hiddenLayerSize = [80 40 40 20];
% hiddenLayerSize = [10];
net = patternnet(hiddenLayerSize);
% net = newpnn(hiddenLayerSize);


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;


% Train the Network
[net,tr] = train(net,train_data,labels);
% clear train_data labels
toc

% View the Network
% view(net)

N = 1000;
% load('images_test.mat');
load('labels_test.mat');
labels_test = labels_test(1:N,:)';

%  test_data = zeros(size(images_test,1)*size(images_test,2),size(images_test,3));
%   
%  for i=1:size(images_test,3)
%      for j=1:size(images_test,1)
%        test_data(size(images_test,2)*(j-1)+1:size(images_test,2)*j,i) = images_test(j,:,i);
%      end
%  end
% clear images_test

% test_data = test_data(:,1:N);
% 
% % Neighboors
% K=18;
% % Dimensions
% d=10;
% test_data = lle(test_data,K,d);

load('test_data.mat');
test_data = test_data(:,1:N);
 
% Test the Network
test = net(test_data);
ttind = vec2ind(labels_test);
ytind = vec2ind(test);
testPercentErrors = sum(ttind ~= ytind)/numel(ttind);
fprintf('Test_set percent error %f \n',(testPercentErrors)*100);
performance = perform(net,labels_test,test);

