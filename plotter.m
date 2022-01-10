%% Pupil Position Plotter
close all
data = readmatrix('/Users/Brian/Documents/ENGS89.90/90/Recordings/pupil_positions.csv','NumHeaderLines',1);
data(data(:, 4) <= .65, :)= [];
lefteye = data(data(:,3) == 1, :);
%data = rmoutliers(data);


time = data(:,1)';
x = data(:,5)';
y = data(:,6)';

figure;
hold on;
plot(x,y,'.');

%% Classifier
%max(x);
min(x);
max(y);
min(y);

y_thresh = linspace(min(y), max(y), 4)
x_thresh = linspace(min(x), max(x), 4)
%% Plot
close all;
figure;
hold on;
plot(x,y,'.');
plot([1 1]*x_thresh(1), ylim, '--k') 
plot([1 1]*x_thresh(2), ylim, '--k')                
plot([1 1]*x_thresh(3), ylim, '--k')
plot([1 1]*x_thresh(4), ylim, '--k')
plot(xlim, [1 1]*y_thresh(1), '--k')  
plot(xlim, [1 1]*y_thresh(2), '--k')                
plot(xlim, [1 1]*y_thresh(3), '--k')
plot(xlim, [1 1]*y_thresh(4), '--k')

figure;
plot(linspace(0,1, length(x)),x);
figure;
plot(linspace(0,1, length(y)),y);

%% Calibrate
ur = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/up right.csv','NumHeaderLines',1);
ur(ur(:, 4) <= .65, :)= [];
ur = rmoutliers(ur);

dur_x = ur(:,5)';
dur_y = ur(:,6)';

ul = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/up left.csv','NumHeaderLines',1);
ul(ul(:, 4) <= .65, :)= [];
ul = rmoutliers(ul);

dul_x = ul(:,5)';
dul_y = ul(:,6)';

dr = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/down right.csv','NumHeaderLines',1);
dr(dr(:, 4) <= .65, :)= [];
dr = rmoutliers(dr);

ddr_x = dr(:,5)';
ddr_y = dr(:,6)';

dl = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/down left.csv','NumHeaderLines',1);
dl(dl(:, 4) <= .65, :)= [];
dl = rmoutliers(dl);

ddl_x = dl(:,5)';
ddl_y = dl(:,6)';

c = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/center.csv','NumHeaderLines',1);
c(c(:, 4) <= .65, :)= [];
c = rmoutliers(c);

dc_x = c(:,5)';
dc_y = c(:,6)';

cu = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/center up.csv','NumHeaderLines',1);
cu(cu(:, 4) <= .65, :)= [];
cu = rmoutliers(cu);

dcu_x = cu(:,5)';
dcu_y = cu(:,6)';

cd = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/center down.csv','NumHeaderLines',1);
cd(cd(:, 4) <= .65, :)= [];
cd = rmoutliers(cd);

dcd_x = cd(:,5)';
dcd_y = cd(:,6)';

cr = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/center right.csv','NumHeaderLines',1);
cr(cr(:, 4) <= .65, :)= [];
cr = rmoutliers(cr);

dcr_x = cr(:,5)';
dcr_y = cr(:,6)';

cl = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate2/center left.csv','NumHeaderLines',1);
cl(cl(:, 4) <= .65, :)= [];
cl = rmoutliers(cl);

dcl_x = cl(:,5)';
dcl_y = cl(:,6)';

%%
figure;
hold on;
plot(dcu_x,dcu_y,'.');
plot(dcd_x,dcd_y,'.');
plot(dcr_x,dcr_y,'.');
plot(dcl_x,dcl_y,'.');
plot(dur_x,dur_y,'.');
plot(dul_x,dul_y,'.');
plot(ddr_x,ddr_y,'.');
plot(ddl_x,ddl_y,'.');
plot(dc_x,dc_y,'.');
%%

hold on;
plot(mean(dcu_x), mean(dcu_y), '*');
plot(mean(dcd_x), mean(dcd_y), '*');
plot(mean(dcr_x), mean(dcr_y), '*');
plot(mean(dcl_x), mean(dcl_y), '*');
plot(mean(dc_x), mean(dc_y), '*');
plot(mean(dur_x), mean(dur_y), '*');
plot(mean(dul_x), mean(dul_y), '*');
plot(mean(ddr_x), mean(ddr_y), '*');
plot(mean(ddl_x), mean(ddl_y), '*');

%%
centroids = [[mean(dur_x), mean(dur_y)]; [mean(dcu_x), mean(dcu_y)]; [mean(dul_x), mean(dul_y)]; [mean(dcr_x), mean(dcr_y)]; [mean(dc_x), mean(dc_y)]; [mean(dcl_x), mean(dcl_y)]; [mean(ddr_x), mean(ddr_y)]; [mean(dcd_x), mean(dcd_y)]; [mean(ddl_x), mean(ddl_y)]]
vals = classify([x(:), y(:)], centroids);


%%

d1 = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate3/1.csv','NumHeaderLines',1);
d1(d1(:, 4) <= .65, :)= [];
d1 = rmoutliers(d1);

d1_x = d1(:,5)';
d1_y = d1(:,6)';

d2 = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate3/2.csv','NumHeaderLines',1);
d2(d2(:, 4) <= .65, :)= [];
d2 = rmoutliers(d2);

d2_x = d2(:,5)';
d2_y = d2(:,6)';

d3 = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate3/3.csv','NumHeaderLines',1);
d3(d3(:, 4) <= .65, :)= [];
d3 = rmoutliers(d3);

d3_x = d3(:,5)';
d3_y = d3(:,6)';

d4 = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate3/4.csv','NumHeaderLines',1);
d4(d4(:, 4) <= .65, :)= [];
d4 = rmoutliers(d4);

d4_x = d4(:,5)';
d4_y = d4(:,6)';

d5 = readmatrix('/Users/davidvonderheide/Desktop/Engs 90/Calibrate3/5.csv','NumHeaderLines',1);
d5(d5(:, 4) <= .65, :)= [];
d5 = rmoutliers(d5);

d5_x = d5(:,5)';
d5_y = d5(:,6)';

%figure;
%%
hold on;
plot(d1_x,d1_y,'.');
plot(d2_x,d2_y,'.');
plot(d3_x,d3_y,'.');
plot(d4_x,d4_y,'.');
plot(d5_x,d5_y,'.');

%%
function [x] = classify(data, centroids)
    x = []
    for i = 1:length(data)
        val = 0;
        distances = [];
        for j = 1:length(centroids)
            distances(j) = sqrt(sum(bsxfun(@minus, data(i,:), centroids(j,:)).^2,2));
        end
        [M, I] = min(distances)
        x(i) = I;
    end
end