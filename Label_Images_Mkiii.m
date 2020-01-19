%% ORC 
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' start image labelling'));
detFaceCart = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
detFaceLBP = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
detFaceProfile = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
detUpperBody = vision.CascadeObjectDetector('ClassificationModel','UpperBody');
aaTotal = 0;
aaLabelled = 0;
aaErrors = 0;

%% read all image files
[codeRoot, imageRoot] = fct_projectPath;
pathSource = fullfile(imageRoot, 'OCR','Error','*.j*');
pathTarget = fullfile(imageRoot, 'OCR', 'Labelled_JPEG');
imgList = dir(pathSource)';

% fName = fullfile(imageRoot, 'OCR','Error', 'IMG_20190128_202321.jpg');
% [outputOCR, imgOCR] = fct_ocr(fName);
% figure; imshow(imgOCR); title(iFile.name);

%% loop at the list
for iFile = imgList
    zzzDebug = iFile.name;
    fName = fullfile(iFile.folder, iFile.name);
    imgIn = imread(fName);
    disp(strcat(datestr(now,'HH:MM:SS'),'_',iFile.name));
    [ocrNumber, ocrConfidence, imgOCR] = fct_img_ocr(imgIn);
    figure; imshow(imgOCR);
end


%%
disp(strcat(datestr(now,'HH:MM'),' done'));