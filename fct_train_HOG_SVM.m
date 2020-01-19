function [svmHog,accTrain,accTest,cMtrxTrain,cMtrxTest] = ...
    fct_train_HOG_SVM(vecSizeHOG, imgsetTrain, imgsetTest)
% INM460: Computer Vision Coursework
% Heiko Maerz
%
% this function trains a HOG SVM classifier
% - input : train dataset, test dataset, HOG feature vector size
% - output: trained classifier,
%           test confusion matrix, test and train accuracies
% this function is called from program TrainClassifiers.m

% clear return variables
clear svmHOG;
accTrain = 0; accTest = 0;
clear cMtrxTrain; clear cMtrxTest;

disp(strcat('__extract HOG feature vector @', datestr(now,'HH:MM:SS')));
% training data: extract HOG features
trainData = zeros(size(imgsetTrain,2)*imgsetTrain(1).Count,vecSizeHOG);
obsCount = 0;
for i=1:size(imgsetTrain,2)
    for j = 1:imgsetTrain(i).Count
        obsCount = obsCount + 1;
        trainData(obsCount,:) = ...
            extractHOGFeatures(read(imgsetTrain(i),j));
        trainLabels{obsCount} = imgsetTrain(i).Description;
    end
end

% test data: extract HOG features
testData = zeros(size(imgsetTest,2)*imgsetTest(1).Count,vecSizeHOG);
obsCount = 0;
for i=1:size(imgsetTest,2)
    for j = 1:imgsetTest(i).Count
        obsCount = obsCount + 1;
        testData(obsCount,:) = ...
            extractHOGFeatures(read(imgsetTest(i),j));
        testLabels{obsCount} = imgsetTest(i).Description;
    end
end

% train the model
disp(strcat('__train HOG SVM @', datestr(now,'HH:MM:SS')));
svmHog = fitcecoc(trainData,trainLabels);

% training accuracy
disp(strcat('__determine training accuracy @', datestr(now,'HH:MM:SS')));
[trainPredict,trainScore] = predict(svmHog,trainData);
cMtrxTrain = confusionmat(trainLabels,trainPredict);
accTrain = sum(diag(cMtrxTrain)) / sum(sum(cMtrxTrain));
disp(strcat('__   training accuracy= ', num2str(floor(accTrain*100)), '%'));

% test accuracy
disp(strcat('__determine test accuracy @', datestr(now,'HH:MM:SS')));
[testPredict,testScore] = predict(svmHog,testData);
cMtrxTest = confusionmat(testLabels,testPredict);
accTest = sum(diag(cMtrxTest)) / sum(sum(cMtrxTest));
disp(strcat('__   test accuracy= ', num2str(floor(accTest*100)), '%'));
end

