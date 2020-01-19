function [ocrNumber, ocrConfidence, imgOCR] = fct_img_ocr(imgIn)
%UNTITLED3 Summary of this function goes here 
%   Detailed explanation goes here
ocrNumber = 999;
ocrConfidence = 0.0;
imgOCR = insertText(imgIn,[2 2],'Error');

try
    % scale to maximum 1000 pixel height
    if size(imgIn,1) > 740 | size(imgIn,1) < 700
        iScale = 720 / size(imgIn,1);
        imgIn = imresize(imgIn, iScale);
    end
    
    imgOCR = imgIn;
    
    % check first, maybe not necessary
    [Y,X,C] = size(imgIn);
    if C>1
        imgIn = rgb2gray(imgIn);
    end
    % convert: colour, video, etc! result must be a grayscale image
    % figure; imshow(IG);
    
    %% find the BLOBs - msurf???
    % get the image
    vBrightness = 1 + int8((sum(sum(imgIn))) / (size(imgIn,1) * size(imgIn,2)));
    imgBin = imgIn<vBrightness;
    blobAn = vision.BlobAnalysis('MaximumCount', 500);
    % regions = detectMSERFeatures(imgRaw);
    %     [regions, cc] = detectMSERFeatures(imgRaw);
    %     figure; imshow(imgRaw); hold on;
    %     plot(regions,'showPixelList',true,'showEllipses',false); hold off;
    % find connected blobs
    [area,centroid,bbox] = step(blobAn, imgBin);
    
    bbox = [bbox (bbox(:,3) .* bbox(:,4))];
    %bbox = sortrows(bbox, 5, 'descend');
    bbox = sortrows(bbox, 1, 'ascend');
    
%     imgOCR = im2uint8(imgIn>vBrightness);
%     bCount = 1;
%     for i = 1:size(bbox,1)
%         imgOCR = insertObjectAnnotation(imgOCR,'rectangle',bbox(i,1:4), num2str(bCount));
%         bCount = bCount + 1;
%     end
    
    oBlobs = zeros(size(bbox,1),6);
    imgOCR = im2uint8(imgIn>vBrightness);
    bCount = 1;
    for i = 1:size(bbox,1)
        bbX=bbox(i,1); bbY=bbox(i,2);bbW=bbox(i,3);bbH=bbox(i,4);bbA=bbox(i,5);
        if bbH>bbW     ...
                & (bbH/bbW)<4.2   ...
                & bbH>8           ...
                & bbW>4           ...
                & bbA>100          ...
                & bbA<1000        ...
                & bbX>8           ...
                & (bbX+bbW)<(X-8) ...
                & bbY>8           ...
                & (bbY+bbH)<(Y-8)
            imgOCR = insertObjectAnnotation(imgOCR,'rectangle',bbox(i,1:4), num2str(bCount));
            oBlobs(bCount,1)=floor(bbX+bbW/2);
            oBlobs(bCount,2)=floor(bbY+bbH/2);
            oBlobs(bCount,3)=max(1,bbX-4);
            oBlobs(bCount,4)=max(1,bbY-4);
            oBlobs(bCount,5)=bbW+8;
            oBlobs(bCount,6)=bbH+8;
            bCount = bCount + 1;
        end
    end
    %figure; imshow(imgOCR); title('all boxes');
    
    oBlobs = oBlobs((oBlobs(:,1)>0),:);
    if size(oBlobs,1) == 0
        return;
    elseif size(oBlobs,1) == 0
        oBoxes(1,:) = oBlobs(1,3:6);
    else
        oBlobs = sortrows(oBlobs, 1, 'ascend');
        oBoxes = zeros(size(oBlobs,1),4);
        bCount = 1; merged = false;
        for i = 1:size(oBlobs,1)
            if merged == true
                merged = false;
                continue;
            end
            for j = i:size(oBlobs,1)
                if i == j
                    continue;
                end
                % Verbose for debugging
                % adjacent boxes for numbers:
                % left centroid + box width (left') should be within the right box
                % move left centroid by box width
                zzl_chX = oBlobs(i,1) + oBlobs(i,5); zzl_chY = oBlobs(i,2);
                zzr_X0 = oBlobs(j,3); zzr_X1 = oBlobs(j,3) + oBlobs(j,5);
                zzr_Y0 = oBlobs(j,4); zzr_Y1 = oBlobs(j,4) + oBlobs(j,6);
                zzl_H = oBlobs(i,6); zzr_H = oBlobs(j,6);
                zzh_Min = floor(.9*zzl_H)-1;
                zzh_Max = ceil(1.1*zzl_H)+1;
                if zzr_X0 < zzl_chX .... %bX < (aCX+aW) ...
                        & zzl_chX < zzr_X1 ... % (aCX+aW) < (bX+bW) ...
                        & zzr_Y0 < zzl_chY ... %bY < aCY & aCY < (bY+bH) ....
                        & zzl_chY < zzr_Y1 ...
                        & zzh_Min < zzr_H ...
                        & zzr_H < zzh_Max
                    merged = true;
                    oBoxes(bCount,1) = oBlobs(i,3);
                    oBoxes(bCount,2) = min(oBlobs(i,4), oBlobs(j,4));
                    oBoxes(bCount,3) = (oBlobs(j,3)+oBlobs(j,5)-oBlobs(i,3));
                    zzy_Min = min(oBlobs(i,4), oBlobs(j,4));
                    zzy_Max = max((oBlobs(i,4)+oBlobs(i,6)), (oBlobs(j,4)+oBlobs(j,6)));
                    oBoxes(bCount,4) = zzy_Max - zzy_Min; % this needs to be changed to max (X0-X1)
                    bCount = bCount + 1;
                    break;
                end
            end
            if merged==false
                % add this bb oBlobs(
                % NOTE: check for the very last one as well!!!!
                oBoxes(bCount,:) = oBlobs(i,3:6);
                bCount = bCount + 1;
            end
            
        end
    end
    
    oBoxes = oBoxes((oBoxes(:,1)>0),:);
    imgOCR = im2uint8(imgIn>vBrightness);
    bCount = 1;
    for i = 1:size(oBoxes,1)
        imgOCR = insertObjectAnnotation(imgOCR,'rectangle',oBoxes(i,1:4), num2str(bCount));
        bCount = bCount + 1;
    end
    
    % and get the result
    ocrResult = ocr(imgBin,oBoxes,'CharacterSet','0123456789','TextLayout','Word'); %,'TextLayout','Line');
    % keep only results which returned a number
    ocrMatrix = [0.1 999]; % error value 999 with threshold confidence 0.1
    for i = 1:size(ocrResult,1)
        if ~(isnan(str2double(ocrResult(i,1).Text)))
            ocrMatrix = [ocrMatrix; ocrResult(i,1).WordConfidences str2double(ocrResult(i,1).Text)];
        end
    end
    [~, idx] = max(ocrMatrix(:,1));
    ocrNumber = ocrMatrix(idx,2);
    ocrConfidence = ocrMatrix(idx,1);
    imgOCR = insertText(imgOCR, [4 4], ...
        strcat(num2str(ocrNumber), '_', num2str(floor(ocrConfidence*100)), '%'), ...
        'FontSize', 32);
catch
    disp('Error');
end
end

