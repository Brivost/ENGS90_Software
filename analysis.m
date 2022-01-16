%% Data analysis
%reads in data
close all
read_data = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Test Data/read.csv','NumHeaderLines',1);
read_data(read_data(:, 4) <= .85, :)= [];
%data = rmoutliers(data);


time = read_data(:,1)';
rx = read_data(:,5)';
ry = read_data(:,6)';

figure;
hold on;
plot(rx,ry,'.');


watch_data = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Test Data/watch.csv','NumHeaderLines',1);
watch_data(watch_data(:, 4) <= .85, :)= [];
%data = rmoutliers(data);


time = watch_data(:,1)';
wx = watch_data(:,5)';
wy = watch_data(:,6)';

figure;
hold on;
plot(wx,wy,'.');


%% Classify
%classifies based on centroid values (calculated in plotter script)
centroids = [[mean(dur_x), mean(dur_y)]; [mean(dcu_x), mean(dcu_y)]; [mean(dul_x), mean(dul_y)]; [mean(dcr_x), mean(dcr_y)]; [mean(dc_x), mean(dc_y)]; [mean(dcl_x), mean(dcl_y)]; [mean(ddr_x), mean(ddr_y)]; [mean(dcd_x), mean(dcd_y)]; [mean(ddl_x), mean(ddl_y)]];

r_vals = classify([rx(:), ry(:)], centroids);
w_vals = classify([wx(:), wy(:)], centroids);

%% Feature Extraction

%Divide into epochs
behaviors = [r_vals, w_vals];
r_epochs = buffer(r_vals,600,300,'nodelay');
w_epochs = buffer(w_vals,600,300,'nodelay');

nfeat = 3;

[d1, d2, d3] = size(r_epochs);
r_feat = zeros(d3, nfeat);

[d1, d2, d3] = size(w_epochs);
w_feat = zeros(d3, nfeat);

%extract features
for i=1:length(r_epochs(1,:))
    r_feat(i,1) = mean(r_epochs(:,i));
    r_feat(i,2) = length(unique((r_epochs(:,i))));
    r_feat(i,3) = nnz(diff((r_epochs(:,i))));
end

for i=1:length(w_epochs(1,:))
    w_feat(i,1) = mean(w_epochs(:,i));
    w_feat(i,2) = length(unique((w_epochs(:,i))));
    w_feat(i,3) = nnz(diff((w_epochs(:,i))));
end

%% Train
p = .7;
N = size(r_feat,1); 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;   
tf = tf(randperm(N));

dataTraining = r_feat(tf,:); 
dataTesting = r_feat(~tf,:);
trainLabels = zeros(size(dataTraining(:,1)));
testLabels = zeros(size(dataTesting(:,1)));

p = .7;
N = size(w_feat,1); 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;   
tf = tf(randperm(N));

dataTraining = [dataTraining; w_feat(tf,:)]; 
dataTesting = [dataTesting; w_feat(~tf,:)];
[s1,s2] = size(w_feat(tf,:));
trainLabels = [trainLabels; ones(s1,1)];
[s1,s2] = size(w_feat(~tf,:));
testLabels = [testLabels; ones(s1,1)];

classifier = fitcdiscr(dataTraining, trainLabels);
labels = predict(classifier , dataTesting);
[x,y,t,auc] = perfcurve(testLabels, labels, 1);
figure
plot(x,y)
%%
function [x] = classify(data, centroids)
    x = []
    for i = 1:length(data)
        val = 0;
        distances = [];
        for j = 1:length(centroids)
            distances(j) = sqrt(sum(bsxfun(@minus, data(i,:), centroids(j,:)).^2,2));
        end
        [M, I] = min(distances);
        x(i) = I;
    end
end