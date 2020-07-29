function [n, m] = noll2idx(noll_index)
% transforms the Noll indexing scheme to the standard notation of Zernike
% polynomials. 

% https://en.wikipedia.org/wiki/Zernike_polynomials
conversion = [0,0;1,1;1,-1;2,0;2,-2;2,2;3,-1;3,1;3,-3;3,3;4,0;4,2;4,-2;4,4;4,-4];
n = conversion(noll_index,1);
m = conversion(noll_index,2);

end