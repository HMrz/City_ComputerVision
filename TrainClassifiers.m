% INM460: Computer Vision Coursework
% Heiko Maerz
%
% This program is a wrapper that will train a number of classifiers via a
% dedicated function and save the classifier itself, additional processing
% data (such as resolution, SURF bags or HOG column size, etc.), and the
% classifier name for future identification in a dedicated .mat file

% ----------------------------------------------------------------------- %
%% init
% general variables
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM:SS'),' train classifiers'));
trainPath = fullfile(pwd(), 'Train');
rng(42); % for reproducibility

% read all training sets and split into train and test
% image set of small (75*75) greyscale images
imgsetSmall = imageSet(fullfile(trainPath, 'Small'),'recursive');
[trainSmall,testSmall] = partition(imgsetSmall,[0.8 0.2]);

% image set of medium (95*95) greyscale images
imgsetMedium = imageSet(fullfile(trainPath, 'Medium'),'recursive');
[trainMedium,testMedium] = partition(imgsetMedium,[0.8 0.2]);

% image set of large (115*115) greyscale images
imgsetLarge = imageSet(fullfile(trainPath, 'Large'),'recursive');
[trainLarge,testLarge] = partition(imgsetLarge,[0.8 0.2]);


% ----------------------------------------------------------------------- %
%% Train HOG
% HOG feature vector size
% code copied from this source:
% https://stackoverflow.com/questions/28410628/calculating-feature-size-of-hog
cellSize = [8 8]; % Size of HOG cell, MatLab default
blockSize = [2 2]; % Number of cells in block, MatLab default
blockOverlap = ceil(blockSize/2);
numBins = 9;
% small training images
trainImgSize = [75 75];
blocksPerImage = floor((trainImgSize./cellSize - blockSize)./(blockSize - blockOverlap) + 1);
hogVectorSizeS = prod([blocksPerImage, blockSize, numBins]);
% medium training images
trainImgSize = [95 95];
blocksPerImage = floor((trainImgSize./cellSize - blockSize)./(blockSize - blockOverlap) + 1);
hogVectorSizeM = prod([blocksPerImage, blockSize, numBins]);
% small large images
trainImgSize = [115 115];
blocksPerImage = floor((trainImgSize./cellSize - blockSize)./(blockSize - blockOverlap) + 1);
hogVectorSizeL = prod([blocksPerImage, blockSize, numBins]);

%% train SVM HOG classifiers for the small dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG SVM for dataset small images'));
% train the model via the function
[svmHogS,accTrainSvmHogS,accTestSvmHogS,cMtrxTrainSvmHogS,cMtrxTestSvmHogS] = ...
    fct_train_HOG_SVM(hogVectorSizeS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_SVM_S.mat');
modelName = 'HOG SVM for dataset small images';
featureType = 'HOG';
classifierName = 'SVM';
hogVectorSize = hogVectorSizeS;
imageSize = [75 75];
imageClassifier = compact(svmHogS);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'imageClassifier');

%% train SVM HOG classifiers for the medium dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG SVM for dataset medium images'));
% train the model via the function
[svmHogM,accTrainSvmHogM,accTestSvmHogM,cMtrxTrainSvmHogM,cMtrxTestSvmHogM] = ...
    fct_train_HOG_SVM(hogVectorSizeM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_SVM_M.mat');
modelName = 'HOG SVM for dataset M images';
featureType = 'HOG';
classifierName = 'SVM';
hogVectorSize = hogVectorSizeM;
imageSize = [95 95];
imageClassifier = compact(svmHogM);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'imageClassifier');

%% train SVM HOG classifiers for the large dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG SVM for dataset large images'));
% train the model via the function
[svmHogL,accTrainSvmHogL,accTestSvmHogL,cMtrxTrainSvmHogL,cMtrxTestSvmHogL] = ...
    fct_train_HOG_SVM(hogVectorSizeL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_SVM_L.mat');
modelName = 'HOG SVM for dataset large images';
featureType = 'HOG';
classifierName = 'SVM';
hogVectorSize = hogVectorSizeL;
imageSize = [115 115];
imageClassifier = compact(svmHogL);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'imageClassifier');

%% train MLP HOG classifier for the small dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100] for dataset small images'));
% train the model via the function
[mlpHog10S,labelLookup,accTrainMlpHog10S,accTestMlpHog10S,cMtrxTrainMlpHog10S,cMtrxTestMlpHog10S] = ...
    fct_train_HOG_MLP([100], hogVectorSizeS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP10S.mat');
modelName = 'HOG MLP[100] for dataset small images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeS;
imageSize = [75 75];
imageClassifier = mlpHog10S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the small dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[200] for dataset small images'));
% train the model via the function
[mlpHog20S,labelLookup,accTrainMlpHog20S,accTestMlpHog20S,cMtrxTrainMlpHog20S,cMtrxTestMlpHog20S] = ...
    fct_train_HOG_MLP([200], hogVectorSizeS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP20S.mat');
modelName = 'HOG MLP[200] for dataset small images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeS;
imageSize = [75 75];
imageClassifier = mlpHog20S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the small dataset,
%  two hidden layers with 100 neurons each
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100 100] for dataset small images'));
% train the model via the function
[mlpHog11S,labelLookup,accTrainMlpHog11S,accTestMlpHog11S,cMtrxTrainMlpHog11S,cMtrxTestMlpHog11S] = ...
    fct_train_HOG_MLP([100 100], hogVectorSizeS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP11S.mat');
modelName = 'HOG MLP[100 100] for dataset small images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeS;
imageSize = [75 75];
imageClassifier = mlpHog11S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the medium dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100] for dataset medium images'));
% train the model via the function
[mlpHog10M,labelLookup,accTrainMlpHog10M,accTestMlpHog10M,cMtrxTrainMlpHog10M,cMtrxTestMlpHog10M] = ...
    fct_train_HOG_MLP([100], hogVectorSizeM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP10M.mat');
modelName = 'HOG MLP[100] for dataset medium images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeM;
imageSize = [95 95];
imageClassifier = mlpHog10M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the medium dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[200] for dataset medium images'));
% train the model via the function
[mlpHog20M,labelLookup,accTrainMlpHog20M,accTestMlpHog20M,cMtrxTrainMlpHog20M,cMtrxTestMlpHog20M] = ...
    fct_train_HOG_MLP([200], hogVectorSizeM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP20M.mat');
modelName = 'HOG MLP[200] for dataset medium images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeM;
imageSize = [95 95];
imageClassifier = mlpHog20M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the medium dataset,
%  two hidden layers with 100 neurons each
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100 100] for dataset medium images'));
% train the model via the function
[mlpHog11M,labelLookup,accTrainMlpHog11M,accTestMlpHog11M,cMtrxTrainMlpHog11M,cMtrxTestMlpHog11M] = ...
    fct_train_HOG_MLP([100 100], hogVectorSizeM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP11M.mat');
modelName = 'HOG MLP[100 100] for dataset medium images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeM;
imageSize = [95 95];
imageClassifier = mlpHog11M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the large dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100] for dataset large images'));
% train the model via the function
[mlpHog10L,labelLookup,accTrainMlpHog10L,accTestMlpHog10L,cMtrxTrainMlpHog10L,cMtrxTestMlpHog10L] = ...
    fct_train_HOG_MLP([100], hogVectorSizeL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP10L.mat');
modelName = 'HOG MLP[100] for dataset large images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeL;
imageSize = [115 115];
imageClassifier = mlpHog10L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

% % train MLP HOG classifier for the large dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[200] for dataset large images'));
% train the model via the function
[mlpHog20L,labelLookup,accTrainMlpHog20L,accTestMlpHog20L,cMtrxTrainMlpHog20L,cMtrxTestMlpHog20L] = ...
    fct_train_HOG_MLP([200], hogVectorSizeL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP20L.mat');
modelName = 'HOG MLP[200] for dataset large images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeL;
imageSize = [115 115];
imageClassifier = mlpHog20L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP HOG classifier for the large dataset,
%  two hidden layers with 100 neurons each
disp(strcat(datestr(now,'HH:MM:SS'),' train HOG MLP[100 100] for dataset large images'));
% train the model via the function
[mlpHog11L,labelLookup,accTrainMlpHog11L,accTestMlpHog11L,cMtrxTrainMlpHog11L,cMtrxTestMlpHog11L] = ...
    fct_train_HOG_MLP([100 100], hogVectorSizeL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'HOG_MLP11L.mat');
modelName = 'HOG MLP[100 100] for dataset large images';
featureType = 'HOG';
classifierName = 'MLP';
hogVectorSize = hogVectorSizeL;
imageSize = [115 115];
imageClassifier = mlpHog11L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'hogVectorSize', 'imageSize', 'labelLookup', 'imageClassifier');



% ----------------------------------------------------------------------- %
%% Train SURF
% https://uk.mathworks.com/help/vision/ref/bagoffeatures.html
% construct bag of visual feature words
disp(strcat(datestr(now,'HH:MM:SS'),' extract SURF bag of features, data set small imgages'));
bagSurfS = bagOfFeatures(imgsetSmall,'VocabularySize',1536,'Verbose',false);
disp(strcat(datestr(now,'HH:MM:SS'),' extract SURF bag of features, data set medium imgages'));
bagSurfM = bagOfFeatures(imgsetMedium,'VocabularySize',1536,'Verbose',false);
disp(strcat(datestr(now,'HH:MM:SS'),' extract SURF bag of features, data set large imgages'));
bagSurfL = bagOfFeatures(imgsetLarge,'VocabularySize',1536,'Verbose',false);
try
    save('SURF_BAGS.mat', 'bagSurfS', 'bagSurfM', 'bagSurfL');
catch
end

%% train SVM SURF classifiers for the small dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF SVM for dataset small images'));
% train the model via the function
[svmSurfS,accTrainSvmSurfS,accTestSvmSurfS,cMtrxTrainSvmSurfS,cMtrxTestSvmSurfS] = ...
    fct_train_SURF_SVM(bagSurfS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_SVM_S.mat');
modelName = 'SURF SVM for dataset small images';
featureType = 'SURF';
classifierName = 'SVM';
bagSURF = bagSurfS;
imageSize = [75 75];
imageClassifier = compact(svmSurfS);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'imageClassifier');

%% train SVM SURF classifiers for the medium dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF SVM for dataset medium images'));
% train the model via the function
[svmSurfM,accTrainSvmSurfM,accTestSvmSurfM,cMtrxTrainSvmSurfM,cMtrxTestSvmSurfM] = ...
    fct_train_SURF_SVM(bagSurfM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_SVM_M.mat');
modelName = 'SURF SVM for dataset medium images';
featureType = 'SURF';
classifierName = 'SVM';
bagSURF = bagSurfM;
imageSize = [95 95];
imageClassifier = compact(svmSurfM);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'imageClassifier');

%% train SVM SURF classifiers for the large dataset
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF SVM for dataset large images'));
% train the model via the function
[svmSurfL,accTrainSvmSurfL,accTestSvmSurfL,cMtrxTrainSvmSurfL,cMtrxTestSvmSurfL] = ...
    fct_train_SURF_SVM(bagSurfL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_SVM_L.mat');
modelName = 'SURF SVM for dataset large images';
featureType = 'SURF';
classifierName = 'SVM';
bagSURF = bagSurfL;
imageSize = [115 115];
imageClassifier = compact(svmSurfL);
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'imageClassifier');


%% train MLP SURF classifier for the small dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100] for dataset small images'));
% train the model via the function
[mlpSurf10S,labelLookup,accTrainMlpSurf10S,accTestMlpSurf10S,cMtrxTrainMlpSurf10S,cMtrxTestMlpSurf10S] = ...
    fct_train_SURF_MLP([100], bagSurfS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP10S.mat');
modelName = 'SURF MLP[100] for dataset small images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfS;
imageSize = [75 75];
imageClassifier = mlpSurf10S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the small dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[200] for dataset small images'));
% train the model via the function
[mlpSurf20S,labelLookup,accTrainMlpSurf20S,accTestMlpSurf20S,cMtrxTrainMlpSurf20S,cMtrxTestMlpSurf20S] = ...
    fct_train_SURF_MLP([200], bagSurfS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP20S.mat');
modelName = 'SURF MLP[200] for dataset small images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfS;
imageSize = [75 75];
imageClassifier = mlpSurf20S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the small dataset,
%  two hidden layers with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100 100] for dataset small images'));
% train the model via the function
[mlpSurf11S,labelLookup,accTrainMlpSurf11S,accTestMlpSurf11S,cMtrxTrainMlpSurf11S,cMtrxTestMlpSurf11S] = ...
    fct_train_SURF_MLP([100 100], bagSurfS, trainSmall, testSmall);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP11S.mat');
modelName = 'SURF MLP[100 100] for dataset small images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfS;
imageSize = [75 75];
imageClassifier = mlpSurf11S;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the medium dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100] for dataset medium images'));
% train the model via the function
[mlpSurf10M,labelLookup,accTrainMlpSurf10M,accTestMlpSurf10M,cMtrxTrainMlpSurf10M,cMtrxTestMlpSurf10M] = ...
    fct_train_SURF_MLP([100], bagSurfM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP10M.mat');
modelName = 'SURF MLP[100] for dataset medium images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfM;
imageSize = [95 95];
imageClassifier = mlpSurf10M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the medium dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[200] for dataset medium images'));
% train the model via the function
[mlpSurf20M,labelLookup,accTrainMlpSurf20M,accTestMlpSurf20M,cMtrxTrainMlpSurf20M,cMtrxTestMlpSurf20M] = ...
    fct_train_SURF_MLP([200], bagSurfM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP20M.mat');
modelName = 'SURF MLP[200] for dataset medium images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfM;
imageSize = [95 95];
imageClassifier = mlpSurf20M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the medium dataset,
%  two hidden layers with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100 100] for dataset medium images'));
% train the model via the function
[mlpSurf11M,labelLookup,accTrainMlpSurf11M,accTestMlpSurf11M,cMtrxTrainMlpSurf11M,cMtrxTestMlpSurf11M] = ...
    fct_train_SURF_MLP([100 100], bagSurfM, trainMedium, testMedium);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP11M.mat');
modelName = 'SURF MLP[100 100] for dataset medium images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfM;
imageSize = [95 95];
imageClassifier = mlpSurf11M;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the large dataset,
%  one hidden layer with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100] for dataset large images'));
% train the model via the function
[mlpSurf10L,labelLookup,accTrainMlpSurf10L,accTestMlpSurf10L,cMtrxTrainMlpSurf10L,cMtrxTestMlpSurf10L] = ...
    fct_train_SURF_MLP([100], bagSurfL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP10L.mat');
modelName = 'SURF MLP[100] for dataset large images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfL;
imageSize = [115 115];
imageClassifier = mlpSurf10L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the large dataset,
%  one hidden layer with 200 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[200] for dataset large images'));
% train the model via the function
[mlpSurf20L,labelLookup,accTrainMlpSurf20L,accTestMlpSurf20L,cMtrxTrainMlpSurf20L,cMtrxTestMlpSurf20L] = ...
    fct_train_SURF_MLP([200], bagSurfL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP20L.mat');
modelName = 'SURF MLP[200] for dataset large images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfL;
imageSize = [115 115];
imageClassifier = mlpSurf20L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');

%% train MLP SURF classifier for the large dataset,
%  two hidden layers with 100 neurons
disp(strcat(datestr(now,'HH:MM:SS'),' train SURF MLP[100 100] for dataset large images'));
% train the model via the function
[mlpSurf11L,labelLookup,accTrainMlpSurf11L,accTestMlpSurf11L,cMtrxTrainMlpSurf11L,cMtrxTestMlpSurf11L] = ...
    fct_train_SURF_MLP([100 100], bagSurfL, trainLarge, testLarge);
% save the model
fName = fullfile(trainPath, 'Classifiers', 'SURF_MLP11L.mat');
modelName = 'SURF MLP[100 100] for dataset large images';
featureType = 'SURF';
classifierName = 'MLP';
bagSURF = bagSurfL;
imageSize = [115 115];
imageClassifier = mlpSurf11L;
save(fName, 'modelName', 'featureType' , 'classifierName', ...
    'bagSURF', 'imageSize', 'labelLookup', 'imageClassifier');



%% Alex-net ------------------------------------------------------------- %
disp(strcat(datestr(now,'HH:MM:SS'),' Alex net'));
imgDSAlex = imageDatastore(fullfile(trainPath, 'Alex'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
% split
[trainAlex, valAlex, testAlex] = splitEachLabel(imgDSAlex,0.7,0.15,0.15,'randomized');
% define image augmentation parameters
% https://uk.mathworks.com/help/deeplearning/ref/imagedataaugmenter.html
imgAugmentAlex = imageDataAugmenter( ...
    'RandRotation',[-8,8], ...
    'RandXShear',[-2 2], ...
    'RandYShear',[-2 2],...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXReflection', true, ...
    'RandYReflection', true);
% augment images
trainAugAlex = augmentedImageDatastore([227 227],trainAlex,...
    'DataAugmentation',imgAugmentAlex);
valAugAlex = augmentedImageDatastore([227 227],valAlex,...
    'DataAugmentation',imgAugmentAlex);
testAugAlex = augmentedImageDatastore([227 227],testAlex);

% Alex-net parameters
disp(strcat('__initialise Alex CNN @', datestr(now,'HH:MM:SS')));
AlexNet = alexnet;
% change fully connected classification layer
transferLayers = AlexNet.Layers(1:end-3);
countLabels = numel(categories(trainAlex.Labels));
layers = [
    transferLayers
    fullyConnectedLayer(countLabels,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% set options
miniBatchSize = 10;
iterationsPerEpoch = floor(numel(trainAlex.Labels)/miniBatchSize);
options = trainingOptions('sgdm',... %'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',8,...
    'InitialLearnRate',1e-4,...
    'Momentum', 0.9, ...
    'Verbose', false, ... %true,...
    'Plots','training-progress',...
    'ValidationData',valAugAlex,... %'ValidationPatience',2,...
    'ExecutionEnvironment', 'auto', ...
    'ValidationFrequency',iterationsPerEpoch);
% train the CNN
disp(strcat('__train Alex CNN @', datestr(now,'HH:MM:SS')));
[AlexNet, trainingInfo] = trainNetwork(trainAugAlex,layers,options);
% train, validation, and test accuracy
disp(strcat('__determine training accuracy @', datestr(now,'HH:MM:SS')));
% training accuracy
accTrainAlex = trainingInfo.TrainingAccuracy(end)/100;
disp(strcat('__   training accuracy= ', num2str(floor(accTrainAlex*100)), '%'));
% validation accuracy
disp(strcat('__determine validation accuracy @', datestr(now,'HH:MM:SS')));
accValAlex = trainingInfo.ValidationAccuracy(trainingInfo.ValidationAccuracy > 0);
accValAlex = accValAlex(end)/100;
disp(strcat('__   validation accuracy= ', num2str(floor(accValAlex*100)), '%'));
% test accuracy
disp(strcat('__determine test accuracy @', datestr(now,'HH:MM:SS')));
testLabelAlex = testAlex.Labels;
testPredictionAlex = classify(AlexNet,testAugAlex);
accTestAlex = sum(testPredictionAlex == testLabelAlex)/numel(testLabelAlex);
disp(strcat('__   test accuracy= ', num2str(floor(accTestAlex*100)), '%'));
% save the model
fName = fullfile(trainPath, 'Classifiers', 'ALEX.mat');
modelName = 'ALEX CNN';
featureType = 'NONE';
classifierName = 'CNN';
imageClassifier = AlexNet;
save(fName, 'modelName', 'featureType' , 'classifierName', 'imageClassifier');


%% done
save('TrainClassifiers.mat');
disp(strcat(datestr(now,'HH:MM'),' done'));