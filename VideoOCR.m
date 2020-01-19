%% ORC 
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' start video labelling'));
% detFaceCart = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% detFaceLBP = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
% detFaceProfile = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
% detUpperBody = vision.CascadeObjectDetector('ClassificationModel','UpperBody');
% aaTotal = 0;
% aaLabelled = 0;
% aaErrors = 0;

%% path
[codeRoot, imageRoot] = fct_projectPath;
pathSource = fullfile(imageRoot, 'Videos','*.m*');
videoList = dir(pathSource)';

%%   read video file
for fName = videoList
    disp(strcat(datestr(now,'HH:MM'),'_',fName.name));
    
    fTokens = split(fName.name, '.');
    
    copyFrom = fullfile(fName.folder, fName.name);
    videoReader = VideoReader(copyFrom);
    vImages = read(videoReader);
    nFrame = 1;
    ocrMatrix = [0.1 999];
    for f = 1:size(vImages, 4)
        vStill = vImages(:,:,:,f);
        vGray = rgb2gray(vStill);
        vBrightness = mean(mean(vGray));
        if vBrightness > 75
            [ocrNumber, ocrConfidence, imgOCR] = fct_img_ocr(vStill);
            if ocrConfidence > .7
                ocrMatrix = [ocrMatrix; ocrConfidence ocrNumber];
                copyTo = strcat(fTokens{1}, '_', upper(fTokens{2}), '_', num2str(nFrame,'%03d'), '.jpeg');
                copyTo = fullfile(fName.folder, copyTo);
                imwrite(imgOCR, copyTo, 'Quality', 100);
                nFrame = nFrame + 1;
            end
        end
    end
    [ocrMode, ocrFreq] = mode(ocrMatrix(:,2));
    disp(strcat('   number: ', num2str(ocrMode), ', p=', num2str(floor(ocrFreq*100)),'% ,', ...
         num2str(nFrame), '/', num2str(f), ' frames'));
end

%% finish
disp(strcat(datestr(now,'HH:MM'),' done'));