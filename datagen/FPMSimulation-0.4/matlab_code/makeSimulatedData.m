function makeSimulatedData(groundTruthAmplitude, groundTruthPhase, outputFile, output, parameterFile, zernikeCoefficients, debugIsActive)
% function makeSimulatedData(groundTruthAmplitude, groundTruthPhase, outputFile, output, parameterFile, zernikeCoefficients, debugIsActive)

%% define simulation parameters and ground truth images
% parameterFile = 'simulationParameters.txt';
% zernikeCoefficients = 'zernikeCoefficientsOnAxis.txt';
% groundTruthAmplitude = 'amplitude.tif';
% groundTruthPhase = 'phase.tif';
% outputFile = 'data.tif';
% outputLog = 'log.txt';
% debugIsActive = false;

% helper functions
APODISATION = 0.1;
F = @(x) fftshift(fft2(ifftshift(tukeywin2(x,APODISATION)))); % Fourier Transform
FT = @(x) fftshift(ifft2(ifftshift(tukeywin2(x,APODISATION)))); % Inverse Fourier Transform

%% load paramters and sample
parameters = readMyParameters(parameterFile);

% Custom parameters to be used during training the neural network
parameters.AMPLITUDE = groundTruthAmplitude;
parameters.PHASE = groundTruthPhase;

%% Add Variation to the parameters
% +- 5% variation

% Using beta distribution
% NA = NA + NA * (0.1 * betarnd(4, 4, 1) - 0.05);
% OBSCURATION = OBSCURATION + OBSCURATION * (0.1 * betarnd(4, 4, 1) - 0.05);
% RELATIVE_LEG_WIDTH = RELATIVE_LEG_WIDTH + RELATIVE_LEG_WIDTH * (0.1 * betarnd(4, 4, 1) - 0.05);

% Using rand
NA = NA + NA * (0.1 * rand(1) - 0.05);
parameters.NA = NA;
OBSCURATION = OBSCURATION + OBSCURATION * (0.1 * rand(1) - 0.05);
parameters.OBSCURATION = OBSCURATION ;
RELATIVE_LEG_WIDTH = RELATIVE_LEG_WIDTH + RELATIVE_LEG_WIDTH * (0.1 * rand(1) - 0.05);
parameters.RELATIVE_LEG_WIDTH = RELATIVE_LEG_WIDTH ;

% WAVELENGTH, PIXELSIZE, IMAGESIZE, FRAMES fixed

% Rest of the Values
vals = min_MAG:step_MAG:max_MAG;  % Discrete range of values for MAGNIFICATION
MAGNIFICATION = vals(randi([1 length(vals)], 1));
parameters.MAGNIFICATION = MAGNIFICATION ;

max_val = max_NA_ILL; min_val = min_NA_ILL; % Continuous range of values for NA_ILLUMINATION
NA_ILLUMINATION = (max_val - min_val) * rand(1) + min_val;
parameters.NA_ILLUMINATION = NA_ILLUMINATION ;

max_val = max_RI; min_val = min_RI; % Continuous range of values for RI
RI = (max_val - min_val) * rand(1) + min_val;
parameters.RI = RI ;

max_val = max_ILL_PHASE_OFF; min_val = min_ILL_PHASE_OFF; % Continuous range of values for ILLUMINATION_PHASE_OFFSET
ILLUMINATION_PHASE_OFFSET = (max_val - min_val) * rand(1) + min_val;
parameters.ILLUMINATION_PHASE_OFFSET = ILLUMINATION_PHASE_OFFSET ;

max_val = max_ILL_OFF; min_val = min_ILL_OFF; % Continuous range of values for ILLUMINATION_OFFCENTER_X/Y
ILLUMINATION_OFFCENTER_X = (max_val - min_val) * rand(1) + min_val;
ILLUMINATION_OFFCENTER_Y = (max_val - min_val) * rand(1) + min_val;
parameters.ILLUMINATION_OFFCENTER_X = ILLUMINATION_OFFCENTER_X ;
parameters.ILLUMINATION_OFFCENTER_Y = ILLUMINATION_OFFCENTER_Y ;

vals = min_ADC:step_ADC:max_ADC;  % Discrete range of values for ADC
ADC = vals(randi([1 length(vals)], 1));
parameters.ADC = ADC;

vals = min_SNR:step_SNR:max_SNR;  % Discrete range of values for SNR
SNR = vals(randi([1 length(vals)], 1));
parameters.SNR = SNR;

vals = min_BG:step_BG:max_BG;  % Discrete range of values for BG
BG = vals(randi([1 length(vals)], 1));
parameters.BG = BG;

max_val = max_OPD; min_val = min_OPD; % Continuous range of values for SAMPLE_OPD
SAMPLE_OPD = (max_val - min_val) * rand(1) + min_val;
parameters.SAMPLE_OPD = SAMPLE_OPD ;
%% End

objectAmplitude = double(imread(groundTruthAmplitude));
objectAmplitude = imresize(objectAmplitude,[IMAGESIZE IMAGESIZE])./max(objectAmplitude(:));
objectPhase = double(imread(groundTruthPhase));
objectPhase = SAMPLE_OPD *(imresize(objectPhase,[IMAGESIZE IMAGESIZE])./max(objectPhase(:)));

object = objectAmplitude.*exp(1i.*objectPhase);

imshowComplex(object, debugIsActive)

%% create illumination
k0x = zeros(1,FRAMES);
k0y = zeros(1,FRAMES);

for i = 1:FRAMES
        k0x(1,i) = (2*pi * NA_ILLUMINATION * RI / WAVELENGTH) *...
            cos(2*pi*(i-1)/FRAMES + ILLUMINATION_PHASE_OFFSET);
        k0y(1,i) =  (2*pi * NA_ILLUMINATION * RI / WAVELENGTH) *...
            sin(2*pi*(i-1)/FRAMES + ILLUMINATION_PHASE_OFFSET);
end

parameters.K0X = k0x;
parameters.K0Y = k0y;
%% create CTF
% normal pupil function
CUTOFF_FREQ = 2*pi * NA * RI / WAVELENGTH;
NYQUIST_FREQ = 2*pi / (2*PIXELSIZE / (RI*MAGNIFICATION));

k = linspace(-NYQUIST_FREQ,NYQUIST_FREQ,IMAGESIZE);
[KX,KY] = meshgrid(k,k);
CTF = ((KX.^2 + KY.^2) < CUTOFF_FREQ^2);

% obscuration of reflective objective

CTF_obscuration = ((KX.^2 + KY.^2) < (OBSCURATION*CUTOFF_FREQ)^2);
CTF_legs = (KX > OBSCURATION*CUTOFF_FREQ) & ...
    (KX < CUTOFF_FREQ) & (abs(KY) < RELATIVE_LEG_WIDTH*CUTOFF_FREQ);
CTF_legs = CTF_legs + ...
    imrotate(CTF_legs,120,'bicubic','crop') + ...
    imrotate(CTF_legs,-120,'bicubic','crop');
CTF = CTF - (CTF_obscuration + CTF_legs);
% CTF = CTF - (CTF_obscuration);


% pupil aberrations (Standard Zernike RMS coefficients from Zemax model)
wavefrontAberration = zeros(size(CTF));
Z = readZernike(zernikeCoefficients);
for i = 1:length(Z)
    wavefrontAberration = wavefrontAberration + Z(i)* ...
        getZernike(IMAGESIZE,CUTOFF_FREQ,NYQUIST_FREQ,i);
end
CTF = CTF.*exp(-1i*(2*pi * wavefrontAberration));
imshowComplex(CTF, debugIsActive)

%% forward model
objectFT = F(object);
data = zeros(IMAGESIZE,IMAGESIZE,FRAMES);

for i = 1:FRAMES
    % oblique illumination
    delta_x = (IMAGESIZE/2)*k0x(1,i)/NYQUIST_FREQ + ILLUMINATION_OFFCENTER_X;
    delta_y = (IMAGESIZE/2)*k0y(1,i)/NYQUIST_FREQ + ILLUMINATION_OFFCENTER_Y;
    CTF_i = FourierShift2D(CTF, [delta_y delta_x]);

    % effect of aberrations onto oblique illumination
    CTF_i = CTF_i * CTF_i(IMAGESIZE/2,IMAGESIZE/2); 
    
    % remove Fourier transform artifacts
    CTF_i(abs(CTF_i) < 0.5) = 0;
    rawFT_i = CTF_i .* objectFT;
    
    % conversion to intensity
    raw_i = abs(FT(rawFT_i)).^2;
    raw_i = uint16(ADC*raw_i/max(raw_i(:)));

    % Poisson noise
    raw_i = double(imnoise(raw_i,'poisson'));

    % Gaussian white noise and background
    raw_i = awgn(raw_i + BG,SNR);

    data(:,:,i) = raw_i;
end

imshowArray(data, debugIsActive)

%% write to disk
writeMyImages(data,outputFile)
% writeMyLog(parameters,outputLog, overwrite)
save(output, '-struct', 'parameters');
