function [ocrNumber, ocrConfidence, frameCount] = fct_label_video(vPath, vFile)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here 
ocrNumber = 999; ocrConfidence = 0; frameCount = 0;

try
    
    copyFrom = fullfile(vPath, vFile);
    videoReader = VideoReader(copyFrom);
    vImages = read(videoReader);
    nFrame = 0;
    ocrMatrix = [0.1 999];
    for f = 1:min(20,size(vImages, 4))
        vStill = vImages(:,:,:,f);
        vGray = rgb2gray(vStill);
        vBrightness = mean(mean(vGray));
        if vBrightness > 75
            nFrame = nFrame + 1;
            [ocrNumber, ocrConfidence, imgOCR] = fct_img_ocr(vStill);
            if ocrConfidence > .7
                ocrMatrix = [ocrMatrix; ocrConfidence ocrNumber];
%                 copyTo = strcat(fTokens{1}, '_', upper(fTokens{2}), '_', num2str(nFrame,'%03d'), '.jpeg');
%                 copyTo = fullfile(fName.folder, copyTo);
%                 imwrite(imgOCR, copyTo, 'Quality', 100);
            end
        end
    end
    [ocrMode, ocrFreq] = mode(ocrMatrix(:,2));
    ocrNumber = ocrMode;
    ocrConfidence = ocrFreq / size(ocrMatrix,1);
    frameCount = nFrame;
    vImages = []; clear vImages; clear videoReader;
catch
    ocrNumber = 999; ocrConfidence = 0; frameCount = 0;
    disp(strcat('Error_', vFile));
end
end

