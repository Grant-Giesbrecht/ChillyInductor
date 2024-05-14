L_tot = 500e-9;
C_tot = 64.6e-12;
N = 140;

f = 10e9;

displ("OG ", N, " Stages:");

ZL_ = Zl(L_tot/N, f);
ZC_ = Zc(C_tot/N, f);

displ("  ZL_ = ", ZL_ );
displ("  ZC_ = ", ZC_ );
displ("  N = ", N );

Ztot_all = ones(1, N);

idx = 1;
Ztot = ZL_+25; % C_ is shorted for last segment
Ztot_all(idx) = Ztot;

for n = 1:N-1
	idx = idx + 1;
	
% 	displ("    Ztot = ", Ztot);
	
	Ztot = pall(Ztot, ZC_);
	Ztot = Ztot + ZL_;
	
	Ztot_all(idx) = Ztot;
end

X = imag(Ztot);
L = X/2/pi/f;


displ("Z = ", Ztot);
displ("L = ", L*1e9, " nH"); 

figure(2);
hold off;
plot(imag(Ztot_all));
% hold on;
% plot(real(Zta));

figure(5);
hold off;
plotsc(Ztot_all, 'Domain', 'Z', 'LineWidth', 3);

