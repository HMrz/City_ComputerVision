%% ORC 
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' start video label and copy'));
% detFaceCart = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% detFaceLBP = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
% detFaceProfile = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
% detUpperBody = vision.CascadeObjectDetector('ClassificationModel','UpperBody');
% aaTotal = 0;
% aaLabelled = 0;
% aaErrors = 0;

% Files and frames
tblFrames = table('Size', [0, 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'vFile', 'Label', 'Confidence', 'Frames'});

% %% read all image files
% [codeRoot, imageRoot] = fct_projectPath;
% pathSource = fullfile(imageRoot, 'OCR','Error','*.j*');
% pathTarget = fullfile(imageRoot, 'OCR', 'Labelled_JPEG');

%% read all image files
workingDir = pwd();
pathSource = fullfile(workingDir, 'All_Videos','I*.m*');
pathTarget = fullfile(workingDir, 'Labelled_Videos');
videoList = dir(pathSource)';

% fName = fullfile(imageRoot, 'OCR','Error', 'IMG_20190128_202321.jpg');
% [outputOCR, imgOCR] = fct_ocr(fName);
% figure; imshow(imgOCR); title(iFile.name);

%% loop at the list
try
for vFile = videoList
    clear copyFrom; clear copyTo;
    zzzDebug = vFile.name;
    copyFrom = fullfile(vFile.folder, vFile.name);
    disp(strcat(datestr(now,'HH:MM:SS '),'_',vFile.name));
    %[ocrNumber, ocrConfidence, imgOCR] = fct_ocr(copyFrom);
    [ocrNumber, ocrConfidence, frameCount] = fct_label_video(vFile.folder, vFile.name);
%     if frameCount < 50
%         fnLowFrame = fullfile(pathTarget, 'LT50', vFile.name);
%         copyStatus = copyfile(copyFrom, fnLowFrame);
%     end
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
    copyTo = fullfile(copyTo, vFile.name);
    copyStatus = movefile(copyFrom, copyTo);
    if copyStatus < 1
        disp(strcat('ERR  :', vFile.name));
        copyTo = fullfile(pathTarget, 'Error', vFile.name);
        copyStatus = movefile(copyFrom, copyTo);
    end
    rowFrames = {vFile.name, ocrNumber, ocrConfidence, frameCount};
    tblFrames = [tblFrames; rowFrames];
    fnCnt = strcat('fr_', datestr(now,'yyyymmdd_HHMMSS_FFF'), '.mat');
    % save(fnCnt, 'tblFrames');
    %save('vfile_fcount.mat', 'rowFrames');
    %figure; imshow(imgOCR);
end
catch
    disp('Fatal error');
end


%% done
save('label_videos.mat', 'tblFrames');
disp(strcat(datestr(now,'HH:MM'),' done'));