function fct_extractVideoFrames(currWorkingDir,sourceDir,targetDir)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
sourcePath = fullfile(currWorkingDir, sourceDir, '*.m*');
targetPath = fullfile(currWorkingDir, targetDir);
lineOut = strcat({'  '}, 'reading from:', {' '}, sourcePath);
disp(lineOut);
lineOut = strcat({'  '}, 'writing to  :', {' '}, targetPath);
disp(lineOut);
mFiles = dir(sourcePath)';
for fName = mFiles
    fTokens = split(fName.name, '.');
    try
        videoReader = VideoReader(fullfile(fName.folder, fName.name));
        vImages = read(videoReader);
        vFrames = size(vImages, 4);
        for f = 1:vFrames
            vStill = vImages(:,:,:,f);
            imgFName = strcat(upper(fTokens{2}), '_', fTokens{1}, '_', sprintf('%03d', f), '.jpeg');
            %figure; imshow(vStill)
            % is it just all black?
            vGray = rgb2gray(vStill);
            vBrightness = (sum(sum(vGray))) / (size(vGray,1) * size(vGray,2));
            if vBrightness < 80
                lineOut = string(strcat({'  - no img '}, imgFName));
                disp(lineOut);
            else
                fWrite = fullfile(targetPath, imgFName);
                imwrite(vStill, fWrite);
                lineOut = string(strcat({'  + '}, imgFName));
                disp(lineOut);
            end
        end
    catch
        lineOut = strcat('ERROR:_', string(fName.name));
        disp(lineOut);
    end
end
end

