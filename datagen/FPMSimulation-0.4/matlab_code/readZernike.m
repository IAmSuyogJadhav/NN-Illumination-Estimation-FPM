function Z = readZernike(filePath)

if isfile(filePath)

    fileID = fopen(filePath,'r');
    formatSpec = '%f';
    Z = fscanf(fileID,formatSpec);
    fclose(fileID);
    
else
    disp([filePath ' not found. No pupil aberrations generated.']);
    Z = 0;
end

end