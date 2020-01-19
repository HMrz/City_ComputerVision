% INM460: Computer Vision Coursework
% Heiko Maerz
%
% This program contains two parts:
% 1) prepare 'ground truth':
%    - load a test picture, extract faces, save the bounding boxes
%    - the bounding boxes .csv file will be manually labelled and serves
%      as the base for part 2: test classifiers
%    NOTE: part one will be run once and then commented
%
% 2) load training image and .csv file containing bounding boxes and labels
%    obtained from part 1 of this program
%    runs the face classifier for the test image and compares it with
%    ground truth
%    the result is test accuracy, i.e. the percentage of faces detected for
%    each tested classifier

%% init
clear all; close all; clc; warning('off','all');
disp(strcat(datestr(now,'HH:MM:SS'),' test face classifiers'));
tName = 'Test';

%% PART 1: run once per test image and then comment
%  NOTE: run part 1a and part 1b separatedly
% %% PART 1a) - load test image,
% %           - run the face detector,
% %           - download bounding boxes.
% disp(strcat(datestr(now,'HH:MM:SS'),' instantiate detector'));
% % Init CART face detector
% mergeThreshold = 8;
% % %minSize = [200 200]; maxSize = [600 600];
% faceDetectorCART = ...
%     vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART', ...
%     'MergeThreshold', mergeThreshold);
% %     'MinSize', minSize, ...
% %     'MaxSize', maxSize);
% % load test image
% disp(strcat(datestr(now,'HH:MM:SS'),' read face'));
% fnImg = strcat(tName, '.jpg');
% imgGT = imread(fullfile(pwd(), 'Test', fnImg));
% disp(strcat(datestr(now,'HH:MM:SS'),' find faces'));
% % detect faces, annotate test image
% bBox = faceDetectorCART(imgGT);
% for i = 1:size(bBox,1)
%     imgGT = insertObjectAnnotation(imgGT,'rectangle', ...
%         bBox(i,:), num2str(i), ...
%         'FontSize', 30 );
% end
% % save bounding box to manually add labels
% fnBBox = strcat(tName, '.csv');
% fName = fullfile(pwd(), fnBBox);
% csvwrite(fName, bBox);
%
% %% PART 1b) - load new bounding box .csv file containing these columns:
% %                 label, xMin, yMin, width, height
% %          - annotate each extracted face,
% %            annotate with label read in from the .csv file,
% %          - show each annotate face
% %          - annotate the entire test picture and display
% %          - repeat this process until all faces are manually labelled
% % load test image and class label .csv
% fnImg = strcat(tName, '.jpg');
% imgGT = imread(fullfile(pwd(), fnImg));
% fnBBox = strcat(tName, '.csv');
% bBox = csvread(fullfile(pwd(), fnBBox));
% % display each individual face with it's label to ease manual labelling
% for i = 1:size(bBox,1)
%     imgFace = imcrop(imgGT, bBox(i,2:5));
%     imgFace = rgb2gray(imgFace);
%     imgFace = imresize(imgFace, [285 285]);
%     imgFace = insertText(imgFace, [2, 2], num2str(bBox(i,1)), ...
%         'BoxColor', 'white', 'BoxOpacity', 0.4, ... %'black'
%         'FontSize', 48, 'Font', 'LucidaSansDemiBold', ...
%         'TextColor', 'black');
%     figure; imshow(imgFace);
% end
% % annotate and show entire test picture
% for i = 1:size(bBox,1)
%     imgGT = insertObjectAnnotation(imgGT,'rectangle', ...
%         bBox(i,2:5), num2str(bBox(i,1)), ...
%         'FontSize', 12 );
% end
% % save the annotated test picture for future reference
% imwrite(imgGT, fullfile(pwd(), 'Test', 'FullColour.jpg'), 'Quality', 100);
% figure; imshow(imgGT);


%% PART 2: test classifiers
% load test image and class label .csv
fnImg = strcat(tName, '.jpg');
imgTest = imread(fullfile(pwd(), fnImg));
fnBBox = strcat(tName, '.csv');
bBox = csvread(fullfile(pwd(), fnBBox));

% load classifiers
pathClassifiers = fullfile(pwd(), 'Train', 'Classifiers');
mClassifiers = dir(pathClassifiers)';

for classifier = mClassifiers
    if classifier.isdir == true
        continue
    end
    if classifier.name(1) == '.'
        continue
    end
    aaard = classifier.name;
    faceDetectResult = fctTestFaceClassifier(fullfile(classifier.folder, classifier.name), ...
        bBox, imgTest);
    aaard = "";
end
%% finish
disp(strcat(datestr(now,'HH:MM:SS'),' done'));