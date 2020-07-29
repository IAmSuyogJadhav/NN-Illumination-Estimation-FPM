function imshowComplex(image, debugIsActive)

if nargin < 2
    debugIsActive = true;
end

objectAmplitude = abs(image);
objectPhase = angle(image);

if debugIsActive
    figure
    imagesc(objectPhase);
    colormap hsv
    colorbar
    hold on
    showObject(:,:,1) = (objectPhase  + pi) / (2*pi);
    showObject(:,:,2) = 1;
    showObject(:,:,3) = objectAmplitude;
    imshow(hsv2rgb(showObject))
end
    
end