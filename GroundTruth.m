% INM460: Computer Vision Coursework
% Heiko Maerz
%
% quick and dirty: one image for ground truth comparison

%% init
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' Ground Truth'));

% %% generate test group picture
% % run this manually!
% pathSource = fullfile(pwd(), 'I*.j*');
% pathTarget = fullfile(pwd(), 'Test');
% fList = dir(pathSource)';
% i = 1;
% for fName = fList
%     imgTest = imread(fullfile(fName.folder, fName.name));
%     [imdY, imdX, imdC] = size(imgTest);
%     figure; imshow(imgTest);
%     xmin = 1; ymin = 1; width = imdX; height = imdY;
%     imgTest = imcrop(imgTest, [xmin ymin width height]);
%     fOut = strcat('Test_', num2str(i), '.jpg'); i = i + 1;
%     fOut = fullfile(pathTarget, fOut);
%     imwrite(imgTest, fOut, 'Quality', 100);
% end

%% read all image files
pathSource = fullfile(pwd(), 'ExtractFaces', 'Faces_Photos');
pathTarget = fullfile(pwd(), 'Test', 'GroundTruth');

%% read all directories for ground truth
for d = 1:99
    pathClass = fullfile(pathSource, num2str(d));
    pathType = exist(pathClass);
    if (pathType == 7)
        pathClass = fullfile(pathClass, 'I*.j*');
        fList = dir(pathClass)';
        for fName = fList
            imgGT = imread(fullfile(fName.folder, fName.name));
            [imdY, imdX, imdC] = size(imgGT);
            if imdC == 3
                imgGT = rgb2gray(imgGT);
            end
            imgGT = imresize(imgGT, [285, 285]);
            imgGT = insertText(imgGT, [2, 2], num2str(d), ...
                'BoxColor', 'yellow', 'BoxOpacity', 0.85, ... %'black'
                'FontSize', 48, 'Font', 'LucidaSansDemiBold', ...
                'TextColor', 'black');
            fOut = strcat(num2str(d), '.jpg');
            fOut = fullfile(pathTarget, fOut);
            imwrite(imgGT, fOut, 'Quality', 100);
            continue;
        end
    end
end

%% done
disp(strcat(datestr(now,'HH:MM'),' done'));