% Train Data-size
N_train = 60000;
% Test Data-size
N_test = 10000;

load('labels.mat');
load('labels_test.mat');
load('Y.mat');
    
sb = cell(10,1);
for i=1:10
    sb{i,1} = strcat('\color[rgb]{',num2str(1),',',num2str(1),',',num2str(1),'} ');
end
sr = strcat('\color[rgb]{',num2str(1),',',num2str(0),',',num2str(0),'} ');
sg = strcat('\color[rgb]{',num2str(0),',',num2str(1),',',num2str(0),'} ');
sbl = strcat('\color[rgb]{',num2str(0),',',num2str(0),',',num2str(0),'} ');

s1 = cell(10,1);
for i=1:3:9
    s1{i,1} = strcat('\color[rgb]{',num2str(1),',',num2str(0.1*i),',',num2str(0.1*i),'} ');
end
s1{10,1} = strcat('\color[rgb]{',num2str(0),',',num2str(0),',',num2str(0),'} ');

for i=2:3:10
    s1{i,1} = strcat('\color[rgb]{',num2str(0.1*i),',',num2str(1),',',num2str(0.1*i),'} ');
end

for i=3:3:10
    s1{i,1} = strcat('\color[rgb]{',num2str(0.1*i),',',num2str(0.1*i),',',num2str(1),'} ');
end

figure(1);
scatter3(Y(1,:),Y(2,:),Y(3,:),10,'.');
for i=1:N_train
    id_labels = find(labels(i,:) == 1);
    text(Y(1,i),Y(2,i),Y(3,i),strcat(s1{id_labels,1},' ',int2str(mod(id_labels,10))));
end

for i=1:N_test
    id_labels = find(labels_test(i,:) == 1);
    text(Y(1,i+N_train),Y(2,i+N_train),Y(3,i+N_train),strcat(s1{id_labels,1},' ',int2str(mod(id_labels,10))));
end

