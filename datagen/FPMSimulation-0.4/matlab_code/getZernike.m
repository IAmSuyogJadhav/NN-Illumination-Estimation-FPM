function z = getZernike(IMAGESIZE,CUTOFF_FREQ,NYQUIST_FREQ,noll_index)

% some background info on Zernike polynomials to model wavefront
% aberrations:
% https://en.wikipedia.org/wiki/Zernike_polynomials
% https://www.iap.uni-jena.de/iapmedia/de/Lecture/Imaging+and+Aberration+Theory1425078000/IAT14_Imaging+and+Aberration+Theory+Lecture+12+Zernike+polynomials.pdf
% ftp://ftp.bioeng.auckland.ac.nz/jtur044/references/introductory/zernike-wavefront-aberrations.pdf
% https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2018/04/Schwiegerling-Zernike-2018.pdf

% NA_PIXEL is the half-size of the image relative to the cutoff radius
[n, m] = noll2idx(noll_index);
NA_PIXEL = NYQUIST_FREQ/CUTOFF_FREQ;
x = linspace(-NA_PIXEL,NA_PIXEL,IMAGESIZE);

[X,Y] = meshgrid(x,x);
[theta,r] = cart2pol(X,Y);
idx = r<=1;
z = zeros(size(X));
z(idx) = zernfun(n,m,r(idx),theta(idx),'norm');

end

