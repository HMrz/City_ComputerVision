% INM460: Computer Vision Coursework
% Heiko Maerz
%
% Please note: the code is quite verbose to facilitate debugging, because
%              this is a learning experience for me
%
% This programm reads images (either photos or extracted video frames)
% and crops the faces to generate a set of training images
% this process can be repeated with different parameters for the
% CascadeObjectDetector in case that some images do not yield the faces
%
% Input: directory structure:
% Folder     'ExtractFaces'
% Sub-folder 'Labelled_Photos' with subfolders per Target Label: input
% Sub-folder 'Faces_Photos' with subfolders per Target Label: output
% Sub-folder 'Processed_Photos' with subfolders per Target Label: input
% Sub-folder 'Labelled_Videos' with subfolders per Target Lable
% Sub-folder 'Labelled_VFrames' with subfolders per Target Label: input
% Sub-folder 'Faces_VFrames' with subfolders per Target Label: output
% Sub-folder 'Processed_VFrames' with subfolders per Target Label: input
% Sub-folder 'Balanced_Images' with subfolders per Target Label: input
%            this is the basis for the training images, it will contain
%            an equal amount of training images for each class
%
% Folder:    'Train', images used to train the models
% Sub-folder 'Small/Images' with subfolders per Target Label: 75*75 greyscale
% Sub-folder 'Medium/Images' with subfolders per Target Label: 95*95 greyscale
% Sub-folder 'Large/Images' with subfolders per Target Label: 115*115 greyscale
% Sub-folder 'Alex/Images' with subfolders per Target Label: 227*227 RGB


%% init
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM:SS'),' extract training data'));

%% path etc
pathRoot = fullfile(pwd(), 'ExtractFaces');

%% extract faces from photos
disp(strcat(datestr(now,'HH:MM:SS'),' extract faces from photos'));
% initialise the face detectors
% different thresholds
% mergeThreshold = 10;
mergeThreshold = 8;
% mergeThreshold = 4;
% mergeThreshold = 2;

% max and min size for photos reprocessing
% minSize = [120 120]; maxSize = [600 600];
% max and min size for photos
minSize = [200 200]; maxSize = [600 600];

faceDetectorCART = ...
    vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);
faceDetectorLBP = ...
    vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);
faceDetectorProfile = ...
    vision.CascadeObjectDetector('ClassificationModel','ProfileFace', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);

% % reprocess in case of error
% reProcess = [8, 45, 46, 47, 49, 54, 55, 58, 60, 62, 66, 68, 70, 72, 76, 78, 79, 80, 81];
% moveFile = false;
% for d = reProcess

%simplify: run through all possible folders [1:81] and ignore non-existing
for d = 1:99
    moveFile = true;
    try
        pathPhotos = fullfile(pathRoot, 'Labelled_Photos', num2str(d));
        if exist(pathPhotos) == 7
            % folder for faces and for processed photos
            pathFaces = fullfile(pathRoot, 'Faces_Photos', num2str(d));
            pathPProc = fullfile(pathRoot, 'Processed_Photos', num2str(d));
            
            % read all images
            fList = dir(fullfile(pathPhotos, 'I*.j*'))';
            for fName = fList
                success = false;
                imgPhoto = imread(fullfile(fName.folder, fName.name));
                % some images are read 'flipped'
                [imdY, imdX, imC] = size(imgPhoto);
                if imdX > imdY
                    % rotate it
                    disp(strcat('orErr_', num2str(d), '_', fName.name));
                    imgPhoto = imrotate(imgPhoto,270);
                end
                % try CART detector first
                [success,imgFace] = fct_extract_training_face(imgPhoto, ...
                    faceDetectorCART);
                
                % try LBP if no face was found
                if success == false
                    [success,imgFace] = fct_extract_training_face(imgPhoto, ...
                        faceDetectorLBP);
                end
                
                % try profile if no face was found
                if success == false
                    [success,imgFace] = fct_extract_training_face(imgPhoto, ...
                        faceDetectorProfile);
                end
                
                % save the face if one was found, and move original image
                % to 'processed' folder
                if success == true
                    % write face
                    fOut = fullfile(pathFaces, fName.name);
                    imwrite(imgFace, fOut) %, 'Quality', 100);
                    % copy
                    fOut = fullfile(pathPProc, fName.name);
                    if moveFile == true
                        movefile(fullfile(fName.folder, fName.name), ...
                            fullfile(pathPProc, fName.name));
                    end
                else
                    disp(strcat('__no face in photo /', num2str(d), '/', ...
                        fName.name));
                end
            end
        end
    catch
        disp('error extracting faces from photos');
    end
end

%% extract faces from video frames%% initialise the face detectors
disp(strcat(datestr(now,'HH:MM:SS'),' extract faces from videos'));
% different thresholds
% mergeThreshold = 10;
mergeThreshold = 8;
% mergeThreshold = 4;

% max and min size for video frames
minSize = [75 75]; maxSize = [600 600];

faceDetectorCART = ...
    vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);
faceDetectorLBP = ...
    vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);
faceDetectorProfile = ...
    vision.CascadeObjectDetector('ClassificationModel','ProfileFace', ...
    'MergeThreshold', mergeThreshold, ... %);
    'MinSize', minSize, ...
    'MaxSize', maxSize);


%simplify: run through all possible folders [1:81] and ignore non-existing
for d = 1:99
    moveFile = true;
    try
        pathVFrames = fullfile(pathRoot, 'Labelled_VFrames', num2str(d));
        if exist(pathVFrames) == 7
            % folder for faces and for processed video frames
            pathFaces = fullfile(pathRoot, 'Faces_VFrames', num2str(d));
            pathVProc = fullfile(pathRoot, 'Processed_VFrames', num2str(d));
            
            % read all images
            fList = dir(fullfile(pathVFrames, 'I*.j*'))';
            for fName = fList
                success = false;
                
                imgVFrame = imread(fullfile(fName.folder, fName.name));
                % some images are read 'flipped'
                [imdY, imdX, imC] = size(imgVFrame);
                if imdX > imdY
                    % rotate it
                    imgVFrame = imrotate(imgVFrame,270);
                end
                
                % try CART detector first
                [success,imgFace] = fct_extract_training_face(imgVFrame, ...
                    faceDetectorCART);
                
                % try LBP if no face was found
                if success == false
                    [success,imgFace] = fct_extract_training_face(imgVFrame, ...
                        faceDetectorLBP);
                end
                
                % try profile if no face was found
                if success == false
                    [success,imgFace] = fct_extract_training_face(imgVFrame, ...
                        faceDetectorProfile);
                end
                
                % save the face if one was found, and move original image
                % to 'processed' folder
                if success == true
                    % write face
                    fOut = fullfile(pathFaces, fName.name);
                    imwrite(imgFace, fOut) %, 'Quality', 100);
                    % copy
                    fOut = fullfile(pathVProc, fName.name);
                    movefile(fullfile(fName.folder, fName.name), ...
                        fullfile(pathVProc, fName.name));
                else
                    disp(strcat('__no face in video frame /', num2str(d), '/', ...
                        fName.name));
                end
            end
        end
    catch
        disp('error extracting faces from photos');
    end
end

%% balance the training image numbers, split into training and test here?
disp(strcat(datestr(now,'HH:MM:SS'),' balance image sets'));
%% balance photo image sets
try
    % paths
    pathUnbalanced = fullfile(pathRoot, 'Faces_Photos');
    pathBalanced   = fullfile(pathRoot, 'Balanced_Images');
    
    % read the image set
    iSetPhotos = imageSet(pathUnbalanced,'recursive');
    
    % general information for debugging
    labelsPhotos = { iSetPhotos.Description }; % display all labels on one line
    countPhotos = [iSetPhotos.Count]; % show the corresponding count of images
    % the image class with the smallest set of images will determine the images
    % for each class
    minSetCount = min([iSetPhotos.Count]);
    % balance the image sets
    balancedPhotos = partition(iSetPhotos, minSetCount, 'randomize');
    imgCount = [balancedPhotos.Count];
    
    % copy images to balanced folder set
    for i = 1:size(balancedPhotos,2)
        label = balancedPhotos(i).Description;
        imageLocations = balancedPhotos(i).ImageLocation;
        for j = 1:size(imageLocations,2)
            copyFrom = imageLocations{j};
            [filepath,name,ext] = fileparts(copyFrom);
            copyTo = strcat(name, ext);
            copyTo = fullfile(pathBalanced, label, copyTo);
            cStatus = copyfile(copyFrom, copyTo);
            if cStatus < 1
                disp(strcat('Error_', name));
            end
        end
    end
catch
    disp('error balancing photo sets');
end

%% balance video frame image sets
try
    % paths
    pathUnbalanced = fullfile(pathRoot, 'Faces_VFrames');
    pathBalanced   = fullfile(pathRoot, 'Balanced_Images');
    
    % read the image set
    iSetVFrames = imageSet(pathUnbalanced,'recursive');
    
    % general information for debugging
    labelsVFrames = { iSetVFrames.Description }; % display all labels on one line
    countVFrames = [iSetVFrames.Count]; % show the corresponding count of images
    % the image class with the smallest set of images will determine the images
    % for each class
    minSetCount = min([iSetVFrames.Count]);
    % balance the image sets
    balancedVFrames = partition(iSetVFrames, minSetCount, 'randomize');
    imgCount = [balancedVFrames.Count];
    
    % copy images to balanced folder set
    for i = 1:size(balancedVFrames,2)
        label = balancedVFrames(i).Description;
        imageLocations = balancedVFrames(i).ImageLocation;
        for j = 1:size(imageLocations,2)
            copyFrom = imageLocations{j};
            [filepath,name,ext] = fileparts(copyFrom);
            copyTo = strcat(name, ext);
            copyTo = fullfile(pathBalanced, label, copyTo);
            cStatus = copyfile(copyFrom, copyTo);
            if cStatus < 1
                disp(strcat('Error_', name));
            end
        end
    end
catch
    disp('error balancing photo sets');
end

%% create image training set, greyscale 75 * 75
disp(strcat(datestr(now,'HH:MM:SS'),' create scaled training images'));
try
    % paths
    pathBalanced = fullfile(pathRoot, 'Balanced_Images');
    pathTrainSmall = fullfile(pwd(), 'Train', 'Small');
    pathTrainMedium = fullfile(pwd(), 'Train', 'Medium');
    pathTrainLarge = fullfile(pwd(), 'Train', 'Large');
    pathTrainAlex = fullfile(pwd(), 'Train', 'Alex');
    
    % read the images for each class
    for d = 1:99
        pathSource = fullfile(pathBalanced, num2str(d));
        if exist(pathSource) == 7
            % source images
            fList = dir(fullfile(pathSource, 'I*.j*'))';
            for fName = fList
                try
                    % original image
                    copyFrom = fullfile(fName.folder, fName.name);
                    [filepath,name,ext] = fileparts(copyFrom);
                    imgIn = imread(copyFrom);
                    [imdY, imdX, imdC] = size(imgIn);
                    
                    % small training image
                    imgOut = imresize(imgIn,[75 75]);
                    [outY, outX, outC] = size(imgOut);
                    copyTo = strcat(name, '_BWS', ext);
                    copyTo = fullfile(pathTrainSmall, num2str(d), copyTo);
                    imwrite(imgOut, copyTo, 'Quality', 100);
                    
                    % medium training image
                    imgOut = imresize(imgIn,[95 95]);
                    [outY, outX, outC] = size(imgOut);
                    copyTo = strcat(name, '_BWM', ext);
                    copyTo = fullfile(pathTrainMedium, num2str(d), copyTo);
                    imwrite(imgOut, copyTo, 'Quality', 100);
                    
                    % large training image
                    imgOut = imresize(imgIn,[115 115]);
                    [outY, outX, outC] = size(imgOut);
                    copyTo = strcat(name, '_BWL', ext);
                    copyTo = fullfile(pathTrainLarge, num2str(d), copyTo);
                    imwrite(imgOut, copyTo, 'Quality', 100);
                    
                    % Alex training image
                    imgOut = imresize(imgOut, [227 227]);
                    imgOut = cat(3, imgOut, imgOut, imgOut);
                    [outY, outX, outC] = size(imgOut);
                    copyTo = strcat(name, '_Alex', ext);
                    copyTo = fullfile(pathTrainAlex, num2str(d), copyTo);
                    imwrite(imgOut, copyTo, 'Quality', 100);
                catch
                end
            end
        end
    end
catch
    disp('error scaling training images');
end


%% done
disp(strcat(datestr(now,'HH:MM:SS'),' done'));