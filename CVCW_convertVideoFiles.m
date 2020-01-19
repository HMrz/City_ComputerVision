%% init
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM:SS'),' start reading video directories'));

%% get directory
%imageRoot = pwd();
[codeRoot, imageRoot] = fct_projectPath;
videoDirs = {'MP4', 'MOV'}; %{'testVideo'};

%% read all video files
for dirSource = videoDirs
    dirTarget = strcat(dirSource,  '_converted');
    fct_extractVideoFrames(string(imageRoot), string(dirSource), string(dirTarget));
end

%% done
disp(strcat(datestr(now,'HH:MM:SS'),' done'));