function imshowArray(imageStack, debugIsActive)

if nargin < 2
    debugIsActive = true;
end

if debugIsActive
    figure
    imageCells = num2cell(imageStack,[1 2]);
    FRAMES = size(imageStack,3);
    ARRAYSIZE = ceil(sqrt(FRAMES));
    tiledOut = imtile(imageCells,'Frames', 1:FRAMES, 'GridSize', [ARRAYSIZE ARRAYSIZE]);
    imshow(tiledOut,[])
end
    
end