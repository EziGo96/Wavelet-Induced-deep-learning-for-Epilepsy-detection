clc;clear;close all;
m=load('C:\Users\somsh\OneDrive\Documents\waveletresnet29/weights.mat');
fn = fieldnames(m);
for k=2:numel(fn)
    kernel_k=m.(fn{k}).Conv1D2(:,:,1,1);
    kernel(:,k)=kernel_k;
end
figure;plot(kernel);
figure;
subplot(1,3,1);plot(kernel(:,2),'linewidth',2);
subplot(1,3,2);plot(kernel(:,3),'linewidth',2);
subplot(1,3,3);plot(kernel(:,101),'linewidth',2);
