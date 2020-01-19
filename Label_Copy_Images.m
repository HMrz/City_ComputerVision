%% ORC 
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' start image label and copy'));
% detFaceCart = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% detFaceLBP = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
% detFaceProfile = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
% detUpperBody = vision.CascadeObjectDetector('ClassificationModel','UpperBody');
% aaTotal = 0;
% aaLabelled = 0;
% aaErrors = 0;

% Files and frames
tblLabels = table('Size', [0, 3], ...
    'VariableTypes', {'string', 'double', 'double'}, ...
    'VariableNames', {'vFile', 'Label', 'Confidence'});

% %% read all image files
% [codeRoot, imageRoot] = fct_projectPath;
% pathSource = fullfile(imageRoot, 'OCR','Error','*.j*');
% pathTarget = fullfile(imageRoot, 'OCR', 'Labelled_JPEG');

%% read all image files
workingDir = pwd();
pathSource = fullfile(workingDir, 'All_JPEG','I*.j*');
pathTarget = fullfile(workingDir, 'Labelled_JPEG');
imgList = dir(pathSource)';

% fName = fullfile(imageRoot, 'OCR','Error', 'IMG_20190128_202321.jpg');
% [outputOCR, imgOCR] = fct_ocr(fName);
% figure; imshow(imgOCR); title(iFile.name);

%% loop at the list
for iFile = imgList
    clear copyFrom; clear copyTo;
    zzzDebug = iFile.name;
    copyFrom = fullfile(iFile.folder, iFile.name);
    disp(strcat(datestr(now,'HH:MM:SS'),'_',iFile.name));
    imgIn = imread(copyFrom);
    [ocrNumber, ocrConfidence, imgOCR] = fct_img_ocr(imgIn);
    if ocrNumber > 0 & ocrNumber < 100
        group = num2str(ocrNumber);
        copyTo = fullfile(pathTarget, group);
        pathType = exist(copyTo);
        if ~(pathType == 7)
            copyTo = fullfile(pathTarget, 'Error');
        end
    else
        copyTo = fullfile(pathTarget, 'Error');
    end
    copyTo = fullfile(copyTo, iFile.name);
    copyStatus = copyfile(copyFrom, copyTo);
    if copyStatus < 1
        disp(strcat('ERR  :', iFile.name));
        copyTo = fullfile(pathTarget, 'Error', iFile.name);
        copyStatus = copyfile(copyFrom, copyTo);
    end
    rowLabel = {iFile.name, ocrNumber, ocrConfidence};
    tblLabels = [tblLabels; rowLabel];
    %figure; imshow(imgOCR);
end


%%
save('label_images.mat', 'tblLabels');
disp(strcat(datestr(now,'HH:MM'),' done'));