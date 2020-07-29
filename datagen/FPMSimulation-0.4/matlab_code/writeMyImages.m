function writeMyImages(imageStack,fileName)

FRAMES = size(imageStack,3);

% if isfile(fileName)
%      % File exists.
%      answer = questdlg([fileName ' does already exist. Overwrite?']);

    % Handle response
% switch ancleaswer
%     case 'Yes'
imwrite(uint16(imageStack(:,:,1)), fileName, 'Compression','none');
for i = 2:FRAMES
    imwrite(uint16(imageStack(:,:,i)), fileName, ...
        'WriteMode','append','Compression','none');   
end
%disp([fileName ' was overwritten.']);

%     case 'No'
%         [~, baseFileName, extension] = fileparts(fileName);
%         idx = 0;
%         while isfile([baseFileName '_' num2str(idx) extension])
%             idx = idx + 1;
%         end
% 
%         fileNameNew = [baseFileName '_' num2str(idx) extension];
% 
%         for i = 1:FRAMES
%             imwrite(uint16(imageStack(:,:,i)), fileNameNew, ...
%                 'WriteMode','append','Compression','none');   
%         end
%         disp([fileNameNew ' saved.']);
% 
%     case 'Cancel'
%         disp([fileName ' not saved.']);
% end

% else
%      % File does not yet exist.
%     for i = 1:FRAMES
%         imwrite(uint16(imageStack(:,:,i)), fileName, ...
%             'WriteMode','append','Compression','none');   
%     end
%     disp([fileName ' saved.']);
% end

end