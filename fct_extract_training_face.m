function [rc,imgFace] = fct_extract_training_face(imgSource, detector)
% INM460: Computer Vision Coursework
% Heiko Maerz
%
% this function receives an image and a detector as input and returns one face, 
% if detected by the passed in face detector;
% called from program ExtractTrainingFaces.m

rc = false;
imgFace = imgSource;

try
    % general image data
    [imdY, imdX, imdC] = size(imgSource);
    % convert to greyscale if necessary
    if imdC == 3
        imgSource = rgb2gray(imgSource);
    end
    % is there exactly one frontal face?
    bBox = detector(imgSource);
    if isempty(bBox)
        return;
    end
    % there might be more than one face detected --> find the 'highest'
    % face
    clear idx;
    [~,idx]=max(bBox(:,4));
    if idx > 0
        faceBorder = bBox(idx,:);
        % pad the face a bit
        if faceBorder(1) - 2 > 1 ...
                & faceBorder(2) - 2  > 1 ...
                & (faceBorder(1) + faceBorder(3) +4) < imdX ...
                & (faceBorder(2) + faceBorder(4) +4) < imdY
            faceBorder(1) = max(1, faceBorder(1)-2);
            faceBorder(2) = max(1, faceBorder(2)-2);
            faceBorder(3) = faceBorder(3) + 4;
            faceBorder(4) = faceBorder(4) + 4;
        end
        % just for debugging
        imgDebug = insertObjectAnnotation(imgSource,'rectangle',faceBorder, 'face');
        % found this face, return it
        imgFace = imcrop(imgSource, faceBorder);
        rc = true;
    end
catch
    rc = false;
end

end

