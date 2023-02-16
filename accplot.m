clc;clear;close all;
x=categorical({'MexicanHat-ResNet29','Morlet-ResNet29','Laplace-ResNet29','Gaussian-Resnet29','MexGauss-ResNet29','MorLap-ResNet29','MexMorLap-ResNet29','MorLapGauss-ResNet29','4Wavelet-ResNet29','ResNet29'});
y=[0.85,0.93,0.96,0.83,0.88,0.98,0.96,0.95,0.95,0.87;];
bar(x,y);