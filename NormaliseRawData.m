% INM460: Computer Vision Coursework
% Heiko Maerz
%
% This programm normalises the number of training images per lable / target
% NOTE: the directory structure has to be put in place before running this
% program

%% init
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' start normalise'));

%% get directory structure
workingDir = fullfile(pwd(), 'ExtractFaces');
pathSource = fullfile(workingDir, 'All_Image_Folders', 'Labelled_VFrames') 
pathTarget = fullfile(workingDir, 'Labelled_VFrames');

%% image set
imgSetFaces = imageSet(pathSource,'recursive');

%% normalise the image set

imgCount = [imgSetFaces.Count]; % show the corresponding count of images
minCount = min([imgSetFaces.Count]);
% Notice that each set now has exactly the same number of images.
imgSetFaces = partition(imgSetFaces, minCount, 'randomize');
imgCount = [imgSetFaces.Count];

%% extract and copy
for i = 1:size(imgSetFaces,2)
    label = imgSetFaces(i).Description;
    imageLocations = imgSetFaces(i).ImageLocation;
    for j = 1:size(imageLocations,2)
        copyFrom = imageLocations{j};
        [filepath,name,ext] = fileparts(copyFrom);
        copyTo = strcat(name, ext);
        copyTo = fullfile(pathTarget, label, copyTo);
        cStatus = copyfile(copyFrom, copyTo);
        if cStatus < 1
            disp(strcat('Error_', name));
        end
    end
end

%% done
disp(strcat(datestr(now,'HH:MM'),' done'));