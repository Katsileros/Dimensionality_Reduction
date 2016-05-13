clear all; clc;
load('train_32x32.mat');
train_labels = y;
save('train_labels.mat','train_labels');
data = X;
clear X y train_labels;

% X = zeros(size(data,1)*size(data,2),size(data,4));
X = zeros(size(data,1),size(data,2),size(data,4));
for i=1:size(data,4)
    X(:,:,i) = single(rgb2gray(data(:,:,:,i)));
%     for j=1:size(data,1)
%        X(size(data,2)*(j-1)+1:size(data,2)*j,i) = rgb2gray(data(j,:,:,i)); 
%     end
end
save('X.mat','X');
clear X;

clear all; clc;
load('extra_32x32.mat');
extra_labels = y;
save('extra_labels.mat','extra_labels');
data = X;
clear X y extra_labels;

% X = zeros(size(data,1)*size(data,2),size(data,4));
extraX = zeros(size(data,1),size(data,2),size(data,4));
for i=1:size(data,4)
    extraX(:,:,i) = single(rgb2gray(data(:,:,:,i)));
%     for j=1:size(data,1)
%        X(size(data,2)*(j-1)+1:size(data,2)*j,i) = rgb2gray(data(j,:,:,i)); 
%     end
end
save('extraX.mat','extraX','-v7.3');
clear extraX;

clear all; clc;
load('test_32x32.mat');
test_labels = y;
save('test_labels.mat','test_labels');
data = X;
clear X y test_labels;

% testX = zeros(size(data,1)*size(data,2),size(data,4));
testX = zeros(size(data,1),size(data,2),size(data,4));
for i=1:size(data,4)
    testX(:,:,i) = single(rgb2gray(data(:,:,:,i)));
%     for j=1:size(data,1)
%        testX(size(data,2)*(j-1)+1:size(data,2)*j,i) = rgb2gray(data(j,:,:,i)); 
%     end
end
save('testX.mat','testX');
clear all;

load('extraX.mat');
load('extra_labels.mat');
load('X.mat');
load('train_labels.mat');

train_data = zeros(size(X,1), size(X,2), size(X,3)+size(extraX,3));
labels = zeros(size(train_data,3),1);

for i=1:size(X,3)
   train_data(:,:,i) = X(:,:,i); 
   labels(i,1) = train_labels(i,1);
end

for i=size(X,3)+1:size(extraX,3)
   train_data(:,:,i) = extraX(:,:,i);
   labels(i,1) = extra_labels(i,1);
end

clear X extraX extra_labels train_labels i
X = train_data;
train_labels = labels;
clear train_data labels

save('final_train_data.mat','X','-v7.3');
save('final_train_labels.mat','train_labels','-v7.3');
