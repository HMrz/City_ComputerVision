%% ORC
clear all; close all; clc;
disp(strcat(datestr(now,'HH:MM'),' augment normalised images'));

%% get directory structure
workingDir = pwd();
pathSource = fullfile(workingDir, 'Labelled_JPEG') %'Raw_Images');
pathTarget = fullfile(workingDir, 'Labelled_Augmented');

%% loop over label directories
%errors = [3, 8, 9, 14, 15, 24, 45, 46, 47, 49, 51, 54, 55, 58, 60, 62, 66, 68, 70, 72, 76, 78, 79, 80, 81];
for d = 1:99  %errors %
    try
        labelSource = fullfile(pathSource, num2str(d));
        labelTarget = fullfile(pathTarget, num2str(d));
        if exist(labelSource) == 7 & exist(labelTarget) == 7
            fList = dir(fullfile(labelSource, 'I*.*'))';
            for fName = fList
                copyFrom = fullfile(fName.folder, fName.name);
                copyTo   = fullfile(labelTarget, fName.name);
                cStatus = copyfile(copyFrom, copyTo);
                if cStatus < 1
                    disp(strcat('Error_', num2str(d), '_', fName.name));
                end
                try
                    [filepath,name,ext] = fileparts(copyTo);
                    imgIn = imread(copyFrom);
                    imgOut = flip(imgIn,2);
                    outRot = strcat(name, '_R0', ext);
                    outRot = fullfile(filepath, outRot);
                    imwrite(imgOut, outRot, 'Quality', 100);
                    
                    imgGPU = gpuArray(imgIn);
                    [yO, xO, cO] = size(imgGPU);
                    
                    gpuRot = imrotate(imgGPU,-6,'bilinear','loose');
                    imgRot = gather(gpuRot);
                    [yR, xR, cR] = size(imgRot);
                    outRot = strcat(name, '_R1', ext);
                    outRot = fullfile(filepath, outRot);
                    imwrite(imgRot, outRot, 'Quality', 100);
                    
                    gpuRot = imrotate(imgGPU,-3,'bilinear','loose');
                    imgRot = gather(gpuRot);
                    [yR, xR, cR] = size(imgRot);
                    outRot = strcat(name, '_R2', ext);
                    outRot = fullfile(filepath, outRot);
                    imwrite(imgRot, outRot, 'Quality', 100);
                    
                    gpuRot = imrotate(imgGPU,3,'bilinear','loose');
                    imgRot = gather(gpuRot);
                    [yR, xR, cR] = size(imgRot);
                    outRot = strcat(name, '_R3', ext);
                    outRot = fullfile(filepath, outRot);
                    imwrite(imgRot, outRot, 'Quality', 100);
                    
                    gpuRot = imrotate(imgGPU,6,'bilinear','loose');
                    imgRot = gather(gpuRot);
                    [yR, xR, cR] = size(imgRot);
                    outRot = strcat(name, '_R4', ext);
                    outRot = fullfile(filepath, outRot);
                    imwrite(imgRot, outRot, 'Quality', 100);
                    %                  yC = max(1, floor(yR/2 - yO/2));
                    %                  xC = max(1,floor(xR/2 - xO/2));
                    %                  wC = min(xO, xR);
                    %                  hC = min(yO, yR);
                    %                  imgRCrop = imcrop(imgRot, [xC, yC, wC, hC]);
                catch
                    disp('Error rotating');
                end
            end
        end
    catch
    end
end

%% done
disp(strcat(datestr(now,'HH:MM'),' done'));