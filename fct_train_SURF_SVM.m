function [svmSurf,accTrain,accTest,cMtrxTrain,cMtrxTest] = ...
    fct_train_SURF_SVM(featureBagSurf, imgsetTrain, imgsetTest)
% INM460: Computer Vision Coursework
% Heiko Maerz
%
% this function trains a SURF SVM classifier
% - input : train dataset, test dataset, SURF bag of features
% - output: trained classifier,
%           test confusion matrix, test and train accuracies
% this function is called from program TrainClassifiers.m

% clear return variables
clear svmSURF;
accTrain = 0; accTest = 0;
clear cMtrxTrain; clear cMtrxTest;

disp(strcat('__extract SURF features @', datestr(now,'HH:MM:SS')));
% training data: extract SURF features
trainData = encode(featureBagSurf, imgsetTrain, 'Verbose', false);
labelLookup = { imgsetTrain.Description };
obsCount = 0;
for i=1:size(imgsetTrain,2)
    for j = 1:imgsetTrain(i).Count
        obsCount = obsCount + 1;
        trainLabels{obsCount} = imgsetTrain(i).Description;
    end
end

% test data: extract SURF features
testData = encode(featureBagSurf, imgsetTest, 'Verbose', false);
obsCount = 0;
for i=1:size(imgsetTest,2)
    for j = 1:imgsetTest(i).Count
        obsCount = obsCount + 1;
        testLabels{obsCount} = imgsetTest(i).Description;
    end
end

% train the model
disp(strcat('__train SURF SVM @', datestr(now,'HH:MM:SS')));
svmSurf = fitcecoc(trainData,trainLabels);

% training accuracy
disp(strcat('__determine training accuracy @', datestr(now,'HH:MM:SS')));
[trainPredict,trainScore] = predict(svmSurf,trainData);
cMtrxTrain = confusionmat(trainLabels,trainPredict);
accTrain = sum(diag(cMtrxTrain)) / sum(sum(cMtrxTrain));
disp(strcat('__   training accuracy= ', num2str(floor(accTrain*100)), '%'));

% test accuracy
disp(strcat('__determine test accuracy @', datestr(now,'HH:MM:SS')));
[testPredict,testScore] = predict(svmSurf,testData);
cMtrxTest = confusionmat(testLabels,testPredict);
accTest = sum(diag(cMtrxTest)) / sum(sum(cMtrxTest));
disp(strcat('__   test accuracy= ', num2str(floor(accTest*100)), '%'));
end

