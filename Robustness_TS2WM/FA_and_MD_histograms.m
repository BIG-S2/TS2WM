clear;clc;
%The histograms of mean FA and mean MD (figure 7)

%% FA mean
load('E:\code\TS2WM\Robustness_TS2WM\SFAmean.mat');
load('E:\code\TS2WM\Robustness_TS2WM\FAmean.mat');
X=mean(FA);
X1=mean(SFA);
D1=std(FA);
D2=std(SFA);
X=[X;X1];
D=[D1;D2];
figure;
bar(X,'grouped','barWidth', 1);
hold on
% x=1:28;
xstring={'HCP';'Simulated HCP'}; 
numgroups = size(X, 1);
numbars = size(X, 2);
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);% Aligning error bar with individual bar
errorbar(x, X(:,i), D(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
ylabel('Mean FA');
set(gca,'XTicklabel',xstring,'FontName','Times New Roman','FontSize',14);
set(gca,'XTickLabelRotation',15);

%% MD mean
load('E:\code\TS2WM\Robustness_TS2WM\SMDmean.mat');
load('E:\code\TS2WM\Robustness_TS2WM\MDmean.mat');
X=mean(FA);
X1=mean(SFA);
D1=std(FA);
D2=std(SFA);
X=[X;X1];
D=[D1;D2];
figure;
bar(X,'grouped','barWidth', 1);
hold on
% x=1:28;
xstring={'HCP';'Simulated HCP'}; 
numgroups = size(X, 1);
numbars = size(X, 2);
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);% Aligning error bar with individual bar
errorbar(x, X(:,i), D(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
ylabel('Mean MD');
set(gca,'XTicklabel',xstring,'FontName','Times New Roman','FontSize',14);
set(gca,'XTickLabelRotation',15);