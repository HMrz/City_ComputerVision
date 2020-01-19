%% Convert to b&w
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM:SS'),' start b&w conversion'));

%% get directory
[codeRoot, imageRoot] = fct_projectPath;
dirSource = fullfile(imageRoot, 'All_JPEG', '*.j*');
dirTarget = fullfile(imageRoot, 'All_BW');
dirError  = fullfile(imageRoot, 'Conv_Error');

%% read, convert, and write
mFiles = dir(dirSource)';

for fName = mFiles
    fInput = fullfile(imageRoot, 'All_JPEG', fName.name);
    fBWImg = strcat('BW_', fName.name);
    fBWImg = fullfile(dirTarget, fBWImg);
    imgRGB = imread(fInput);
    imgBW  = rgb2gray(imgRGB);
    imwrite(imgBW, fBWImg);
end

%% and done
disp(strcat(datestr(now,'HH:MM:SS'),' done'));