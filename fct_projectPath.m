function [codeRoot,imageRoot] = fct_projectPath()
% Returns the root directory for the code files and image files 
% assumes a structure
% 'ComputerVision_cw'
% -->CompVis_Code
%    |-->Computer_Vision_ML (codeRoot)
%    |-->TrainingImages     (imageRoot)

codeRoot = '';
imageRoot = '';
idx = 0;

workingDir = pwd();
idx = strfind(workingDir, 'CompVis_Code') - 1;
if idx > 1
    codeRoot = fullfile(workingDir(1:idx), 'CompVis_Code', 'Computer_Vision_ML');
    imageRoot = fullfile(workingDir(1:idx), 'CompVis_Code', 'TrainingImages');
end
end

