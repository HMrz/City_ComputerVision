function [mlpHog,labelLookup,accTrain,accTest,cMtrxTrain,cMtrxTest] = ...
    fct_train_SURF_MLP(mlpLayers, featureBagSurf, imgsetTrain, imgsetTest)
% INM460: Computer Vision Coursework
% Heiko Maerz
%
% this function trains a SURF MLP classifier
% - input : MLP architecture: hidden layers and neurons,
%           train dataset, test dataset, SURF bag of features
% - output: trained classifier,
%           test confusion matrix, test and train accuracies
% this function is called from program TrainClassifiers.m

% clear return variables
clear mlpHog;
accTrain = 0; accTest = 0;
clear cMtrxTrain; clear cMtrxTest; clear labelLookup;

disp(strcat('__extract SURF feature vector @', datestr(now,'HH:MM:SS')));
% training data: extract HOG features
labelLookup = { imgsetTrain.Description };
trainData = encode(featureBagSurf, imgsetTrain, 'Verbose', false);
trainData = double(trainData); %for GPU
% training data: one-hot encoding for the labels
trainTarget = zeros(size(labelLookup,2),size(trainData,1));
vecTarget = zeros(size(trainData,2),1);
obsCount = 0;
for i=1:size(imgsetTrain,2)
    for j = 1:imgsetTrain(i).Count
        obsCount = obsCount + 1;
        obsLabel = imgsetTrain(i).Description;
        labelIdx = find(strcmp(labelLookup, obsLabel));
        vecTarget(obsCount) = str2num(obsLabel);
        trainTarget(labelIdx,obsCount) = 1;
        trainLabels{obsCount} = imgsetTrain(i).Description;
    end
end

% test data: extract HOG features
testData = encode(featureBagSurf, imgsetTest, 'Verbose', false);
testData = double(testData);
% test data: one-hot encoding for the labels
testTarget = zeros(size(labelLookup,2),size(testData,1));
vecTarget = zeros(size(testData,2),1);
obsCount = 0;
for i=1:size(imgsetTest,2)
    for j = 1:imgsetTest(i).Count
        obsCount = obsCount + 1;
        obsLabel = imgsetTest(i).Description;
        labelIdx = find(strcmp(labelLookup, obsLabel));
        vecTarget(obsCount) = str2num(obsLabel);
        testTarget(labelIdx,obsCount) = 1;
        testLabels{obsCount} = imgsetTest(i).Description;
    end
end

% train the model
disp(strcat('__train HOG MLP @', datestr(now,'HH:MM:SS')));
mlpHog = feedforwardnet(mlpLayers, 'trainscg');
mlpHog = configure(mlpHog,trainData',trainTarget);
[mlpHog, MlpTrain] = train(mlpHog,trainData',trainTarget, 'UseGPU', 'yes');

% training accuracy
disp(strcat('__determine training accuracy @', datestr(now,'HH:MM:SS')));
trainPrediction = mlpHog(trainData');
% map the labels
for i = 1: size(trainPrediction,2)
    [~,idx]=max(trainPrediction(:,i));
    trainPredictLabel{i} = labelLookup{idx};
end
cMtrxTrain = confusionmat(trainLabels,trainPredictLabel);
accTrain = sum(diag(cMtrxTrain)) / sum(sum(cMtrxTrain));
disp(strcat('__   training accuracy= ', num2str(floor(accTrain*100)), '%'));

% test accuracy
disp(strcat('__determine test accuracy @', datestr(now,'HH:MM:SS')));
testPrediction = mlpHog(testData');
% map the labels
for i = 1: size(testPrediction,2)
    [~,idx]=max(testPrediction(:,i));
    testPredictLabel{i} = labelLookup{idx};
end
cMtrxTest = confusionmat(testLabels,testPredictLabel);
accTest = sum(diag(cMtrxTest)) / sum(sum(cMtrxTest));
disp(strcat('__   test accuracy= ', num2str(floor(accTest*100)), '%'));
end
