clear;clc;
% connectome patterns in terms of CSA connectivity and fiber count
%figure 6

addpath('circularGraph');
% load data
load('desikan_roi.mat');
load('CSA_networks_simulated.mat');
load('CSA_networks.mat');

c=mean(CSA_networks,3);
count=zeros(size(c));
count(:,23)=c(:,23);
count(23,:)=c(23,:);
count(count<150)=0;

c=mean(CSA_networks3,3);
count3=zeros(size(c));
count3(:,23)=c(:,23);
count3(23,:)=c(23,:);
count3(count3<150)=0;

figure;
myColorMap = lines(length(count));
myLabel=A;
circularGraph(count,'Colormap',myColorMap,'Label',myLabel);
colorbar;
set(findall(gcf,'type','text'),'FontSize',18,'Fontname','Times New Roman');
colorbar('hide')

figure;
myColorMap = lines(length(count3));
myLabel=A;
circularGraph(count3,'Colormap',myColorMap,'Label',myLabel);
colorbar;
set(findall(gcf,'type','text'),'FontSize',18,'Fontname','Times New Roman');
colorbar('hide')