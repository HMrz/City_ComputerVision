% INM460: Computer Vision Coursework
% Heiko Maerz
%
% This program is used to test the RecogniseFace function,
% it reads in group images and passes them on to the classifier

%% init
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM:SS'),' start group detection'));

%% path etc
imagePath = fullfile(pwd(), 'Group');

%% try a few fixed ones first
imgFName = fullfile(imagePath, 'IMG_8239.jpg');

%% run face recognition
[faces] = RecogniseFace(imgFName, 'HOG', 'SVM');

%%finished
disp(strcat(datestr(now,'HH:MM:SS'),' done')); 