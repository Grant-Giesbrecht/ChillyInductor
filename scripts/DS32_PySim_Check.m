%% Configure problem

Vg = 10;

C_ = 150.2e-12;
L_ = 890e-9;
freq = 10e9;
l_phys = 0.5;

Z0 = sqrt(L_/C_);
VP = 1/sqrt(L_*C_);
wavelength = VP/freq;
theta_tot = 2*pi*l_phys/wavelength;

thetaA = theta_tot
thetaB = 0;

Zg = 50;
ZL = 50;

j = complex(0, 1);

%% Solve for IL

M = (ZL.*cos(thetaB) + j.*Z0.*sin(thetaB)).*(cos(thetaA) + j.*Zg./Z0.*sin(thetaA));
N = ( ZL.*j./Z0.*sin(thetaB) + cos(thetaB) ) .* ( j.*Z0.*sin(thetaA) + Zg.*cos(thetaA) );

IL = Vg./(M + N);

%% Solve for Vx and Ix

Vx = IL.*ZL.*cos(thetaB) + IL.*j.*Z0.*sin(thetaB);
Ix = IL.*ZL.*j./Z0.*sin(thetaB) + IL.*cos(thetaB);

%% Solve for Ig

Ig = Vx.*j./Z0.*sin(thetaA) + Ix.*cos(thetaA);

VL = IL*ZL;

%% Solve with P0-method

Zin = xfmr2zin(Z0, ZL, thetaA+thetaB);
Zin_x = xfmr2zin(Z0, ZL, thetaB);

P0 = 1./2.*Vg.^2.*real(Zin)./( (real(Zin) + real(Zg)).^2 + (imag(Zin) + imag(Zg)).^2 ); % From Pozar

thetaX = angle(Zin_x);
Ix_P0based = sqrt(2.*P0./cos(thetaX)./Zin_x);
Vx_P0based = sqrt( 2.*P0.*Zin_x./cos(thetaX) );
IL_P0based = sqrt(2.*P0./cos(0)./ZL );
VL_P0based = IL_P0based*50;

%% PRint reuslts

VI_theta = abs(angle(Vx)-angle(Ix));
Px = 1/2.*abs(Vx).*abs(Ix).*cos(VI_theta);

barprint("ABCD Matrix Based");
displ("IL: ", abs(IL), " A @ ", 180./pi.*angle(IL), " deg");
displ("VL: ", abs(VL), " V @ ", 180./pi.*angle(VL), " deg");
displ("Vx: ", abs(Vx), " V @ ", 180./pi.*angle(Vx), " deg");
displ("Ix: ", abs(Ix), " A @ ", 180./pi.*angle(Ix), " deg");

displ();
displ("Px = ", Px, " W");

displ();
displ("arg(Ix) - arg(IL): ", 180/pi*(angle(Ix)-angle(IL)) ," deg");
displ("arg(Vx) - arg(Ix): ", 180/pi*(angle(Vx)-angle(Ix)) ," deg");

barprint("P0 Based");
displ("IL: ", abs(IL_P0based), " A @ ", 180./pi.*angle(IL_P0based), " deg");
displ("VL: ", abs(VL_P0based), " V @ ", 180./pi.*angle(VL_P0based), " deg");
displ("Vx: ", abs(Vx_P0based), " V @ ", 180./pi.*angle(Vx_P0based), " deg");
displ("Ix: ", abs(Ix_P0based), " A @ ", 180./pi.*angle(Ix_P0based), " deg");

displ();
displ("Px = ", Px, " W");

displ();
displ("arg(Ix) - arg(IL): ", 180/pi*(angle(Ix_P0based)-angle(IL_P0based)) ," deg");
displ("arg(Vx) - arg(Ix): ", 180/pi*(angle(Vx_P0based)-angle(Ix_P0based)) ," deg");

