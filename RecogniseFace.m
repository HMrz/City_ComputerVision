function [rmFaces] = RecogniseFace(imgFName, featureType, classifierName)
% INM460: Computer Vision Coursework
% Heiko Maerz
%
% this function accepts an image filename,
% and these combinations of featureType and classifierName
% ('SVM' 'HOG')
% ('SVM' 'SURF')
% ('MLP', 'HOG')
% ('MLP', 'SURF')
% ('CNN')
% the function will return a matrix with with one row per face detected
% and the columns 'Label' as the number assigned to the training pictures
% and the x and y pixel coordinates of the centre of the face

rmFaces = zeros(1,3);
try
    % initialise path for classifiers
    clfPath = fullfile(pwd(), 'Classifiers');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %   SET this to false
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    annotate = true;
    
    % initialise the face detectors
    faceDetectorCART = ...
        vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART', ...
        'MergeThreshold', 8);
    faceDetectorLBP = ...
        vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP', ...
        'MergeThreshold', 8);
    faceDetectorProfile = ...
        vision.CascadeObjectDetector('ClassificationModel','ProfileFace', ...
        'MergeThreshold', 8);
    
    % load the image and extract the face(s)
    imgFaces = imread(imgFName);
    [imdY, imdX, imdC] = size(imgFaces);
    
    % convert to greyscale if colour
    if imdC == 3
        imgFaces = rgb2gray(imgFaces);
    end
    
    % find the faces
    clear bBox;
    % try CART first
    bBox = faceDetectorCART(imgFaces);
    
    if isempty(bBox)
        % try LBP next
        bBox = faceDetectorLBP(imgFaces);
    end
    
    if isempty(bBox)
        % try profile last
        bBox = faceDetectorProfile(imgFaces);
    end
    
    if isempty(bBox)
        disp('did not detect any faces');
        return;
    end
    
    % initialise the return matrix
    rmFaces = zeros(size(bBox,1),3);
    
    %     % debugging only: annotate faces
    %     bBox = sortrows(bBox, 1, 'ascend');
    %     for i = 1:size(bBox,1)
    %         imgFaces = insertObjectAnnotation(imgFaces,'rectangle', bBox(i,:), num2str(i));
    %     end
    %     figure; imshow(imgFaces);
    
    if strcmp(upper(classifierName),'CNN')
        
    elseif strcmp(upper(classifierName),'MLP')
        
    elseif strcmp(upper(classifierName), 'SVM')
        % load classifier
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NOTE: folder structure for readme and document
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %   CHANGE: not small medium large, but check!!!
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(upper(featureType), 'HOG')
            load(fullfile(clfPath, 'S', 'SVM_HOG.mat'));
            imgSize = imgSizeS; colsHog = colsHogS;
            bBox = sortrows(bBox, 4, 'descend'); %'highest' face first?
            % loop through all the boxes and classify them
            for i = 1:size(bBox,1)
                rmFaces(i,2) = floor(bBox(i,1) + (bBox(i,3)/2));
                rmFaces(i,3) = floor(bBox(i,2) + (bBox(i,4)/2));
                try
                    % extract the face
                    imgDetect = imcrop(imgFaces, bBox(i,:));
                    imgDetect = imresize(imgDetect, imgSize);
                    % extract the features
                    imgFeatures = zeros(1,colsHog);
                    imgFeatures = extractHOGFeatures(imgDetect);
                    % run the classifier
                    predFace = predict(svmHog,imgFeatures);
                    % save the result
                    rmFaces(i,1) = str2double(predFace{1});
                    if annotate
                        imgFaces = insertObjectAnnotation(imgFaces,'rectangle', ...
                                                          bBox(i,:), predFace{1}, ...
                                                          'FontSize', 24 );
                    end
                catch
                end
            end
        elseif strcmp(upper(featureType), 'SURF')
        else
            disp('unknown feature type');
            return;
        end
    else
        disp('unknown classifier');
    end
    
    if annotate 
        [afPath,afName,afExt] = fileparts(imgFName);
        afFName = strcat(afName, afExt);
        afFName = fullfile(pwd(), 'Test', afFName);
        imwrite(imgFaces, afFName, 'Quality', 100);
    end
catch
    disp('uncaught exception in function RecogniseFace');
end
end

