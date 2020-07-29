function g2 = tukeywin2(g, r)
% applies a Tukey window with cosine fraction r.

[M, N] = size(g);
w1 = tukeywin(M,r)';
w2 = tukeywin(N,r)';
w = w1' * w2;
g2 = g .* w;