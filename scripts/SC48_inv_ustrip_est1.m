a = 525e-6;
b = 100e-9;
w = 500e-6;
c = w.*20;
e_0 = 8.854e-12;
e_r = 10;
h = 1e-3;

f0 = 1e9;

%% Calculate Z0 using voodoo formula pirated from Javascript on https://zalophusdokdo.github.io
%

f0X = f0./1e6; % MHz
wX = w./1e-3; % mm
aX = a./1e-3; % mm
bX = b./1e-3; % mm



u = wX./bX;
k1 = (30.666./u).^0.7528;
k2 = exp(-k1);
fu = 6 + (2.*pi-6).*k2;
k3 = aX./bX;
k4 = log(k3);
k5 = 0.5173-0.1515.*k4;
k6 = 0.3092-0.1047*k4;
a1 = k5.^2;
b1 = k6.^2;
k7 = sqrt(e_0);
ef2 = 1 + k3*(a1 - b1 * log(wX./bX))*(k7-1);
ef = ef2*ef2;

k9 = sqrt(1+4./(u*u));
k10 = log(fu./u+k9);

zz = 60*k10./ef2;

Z0 = zz;
displ("Z0 = ", Z0, ", ef = ", ef, ", k = ", 1/ef2, ", l4 = ", 30e10/ef2/(f0X*1e6));

%% Calculate C using voodoo formula listed in "Inverted ustrip and suspended ustrip with anisotropic substrate"
%

pi = 3.1415926535;

eta = sqrt(e_r);

NUM_SUM = 31;

denom_sum = 0;
delta0 = nan;
for n = 1:2:NUM_SUM
	
	Y = e_0 .* ( coth(n.*pi.*b./c) + eta.*eta .* (coth(n.*pi.*a./c).*coth(n.*pi.*h./c)+eta.*eta)./(eta.*eta.*coth(n.*pi.*a./c)+coth(n.*pi.*h./c)) );
	
	gn = (2./n./pi./Y).*(2.*c./n./pi./w).^2;
	Ln = sin(n.*pi.*w./2./c);
	
	Mn = (2.*c./n./pi./w).^3 .* ( 3.*((n.*pi.*w./2./c).^2-2).*cos(n.*pi.*w./2./c) + (n.*pi.*w./2./c).*((n.*pi.*w./2./c).^2-6).*sin(n.*pi.*w./2./c) +6 );
	
	delta = gn*(Ln+Mn).^2;
	denom_sum = denom_sum + delta;
	
	if isnan(delta0)
		delta0 = delta;
	end
	
	displ("  n = ", n, ": Delta = ", delta, " (", delta./delta0.*100, " %), Sum = ", denom_sum);
end

C = 1.5625./denom_sum;

displ("C, with ", NUM_SUM, " terms: ", C./1e-12, " pF");

%% Estimate Vp and L

displ();
barprint("Estimate Using Calculator (Z0) + Paper (C)");

L = Z0.^2 .* C;
Vp = 1./sqrt(L.*C);

displ("L = ", L./1e-9, ' nH');
displ("Vp = ", Vp, " m/s = c/", round(inv(Vp/3e8)));



%% Estimate using sketchy "paper"

displ();
barprint("Estimate Using Sketchy Paper (L) + Paper (C)");

u = w/b;
r = a/b;

L_sketch_nH = 115 + 353 * exp(-u/.364) + 412 * exp(-u/2.6); % nH/m
L_sketch = L_sketch_nH.*1e-9;

Vp_sketch = 1./sqrt(L_sketch.*C);

displ("L = ", L_sketch_nH, ' nH');
displ("Vp = ", Vp_sketch, " m/s = c/", round(inv(Vp_sketch/3e8)*100)/100);
displ("Z0 = ", sqrt(L_sketch/C), ' Ohms');




















