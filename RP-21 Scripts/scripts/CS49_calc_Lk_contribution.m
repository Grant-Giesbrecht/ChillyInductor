% Z0 = 48.71;
% Lk_psq = 15;
% 
% W = 
% 
% Vp = 3e8/3;
% 
% L = Z0/Vp;
% C = 1/Z0/Vp;
% 
% Lk = Lk_psq * 

d = 100e-9;
W = 2.42e-6;
Euni = 9;

K = sqrt(1 + 12*d/W);

Eh = (2*K*Euni - (K-1)) / (K+1);
Eh