clc;clear;close all;
T = readtable('New folder\H0.csv');
figure;plot(T.accuracy);
