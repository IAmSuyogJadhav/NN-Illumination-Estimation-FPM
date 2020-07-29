function writeMyLog(parameters,fileName, overwrite)

% if isfile(fileName)
%    % File exists.
%      answer = questdlg([fileName ' does already exist. Overwrite?']);
%   % Handle response
switch overwrite
    case true
        fileID = fopen(fileName,'w');
        fprintf(fileID,'%s %f\n','AMPLITUDE',parameters.AMPLITUDE);
        fprintf(fileID,'%s %f\n','PHASE',parameters.PHASE);
        fprintf(fileID,'%s %f\n','NA',parameters.NA);
        fprintf(fileID,'%s %f\n','OBSCURATION',parameters.OBSCURATION);
        fprintf(fileID,'%s %f\n','WAVELENGTH',parameters.WAVELENGTH);
        fprintf(fileID,'%s %f\n','PIXELSIZE',parameters.PIXELSIZE);
        fprintf(fileID,'%s %f\n','MAGNIFICATION',parameters.MAGNIFICATION);
        fprintf(fileID,'%s %f\n','NA_ILLUMINATION',parameters.NA_ILLUMINATION);
        fprintf(fileID,'%s %f\n','RI',parameters.RI);
        fclose(fileID);
%         disp([fileName ' was overwritten.']);

    case false
        idx = 0;
        [~, baseFileName, extension] = fileparts(fileName);
        while isfile([baseFileName '_' num2str(idx) extension])
            idx = idx + 1;
        end

        fileNameNew = [baseFileName '_' num2str(idx) extension];

        fileID = fopen(fileNameNew,'w');
        fprintf(fileID,'%s %f\n','AMPLITUDE',parameters.AMPLITUDE);
        fprintf(fileID,'%s %f\n','PHASE',parameters.PHASE);
        fprintf(fileID,'%s %f\n','NA',parameters.NA);
        fprintf(fileID,'%s %f\n','NA',parameters.NA);
        fprintf(fileID,'%s %f\n','OBSCURATION',parameters.OBSCURATION);
        fprintf(fileID,'%s %f\n','WAVELENGTH',parameters.WAVELENGTH);
        fprintf(fileID,'%s %f\n','PIXELSIZE',parameters.PIXELSIZE);
        fprintf(fileID,'%s %f\n','MAGNIFICATION',parameters.MAGNIFICATION);
        fprintf(fileID,'%s %f\n','NA_ILLUMINATION',parameters.NA_ILLUMINATION);
        fprintf(fileID,'%s %f\n','RI',parameters.RI);
        fclose(fileID);
%         disp([fileNameNew ' saved.']);

%     case 'Cancel'
%         disp([fileName ' not saved.']);
end

% else
%      % File does not yet exist.
%     fileID = fopen(fileName,'w');
%     fprintf(fileID,'%s %f\n','NA',parameters.NA);
%     fprintf(fileID,'%s %f\n','OBSCURATION',parameters.OBSCURATION);
%     fprintf(fileID,'%s %f\n','WAVELENGTH',parameters.WAVELENGTH);
%     fprintf(fileID,'%s %f\n','PIXELSIZE',parameters.PIXELSIZE);
%     fprintf(fileID,'%s %f\n','MAGNIFICATION',parameters.MAGNIFICATION);
%     fprintf(fileID,'%s %f\n','NA_ILLUMINATION',parameters.NA_ILLUMINATION);
%     fprintf(fileID,'%s %f\n','RI',parameters.RI);
%     fclose(fileID);
%     disp([fileName ' saved.']);
% end

end