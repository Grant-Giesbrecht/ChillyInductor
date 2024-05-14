%% Configure problem

thetaA_deg = 20;
thetaB_deg = 30;

thetaA = thetaA_deg.*pi./180;
thetaB = thetaB_deg.*pi./180;
Vg = 10;

Zg = 50;
Z0 = 40;
ZL = 30;

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

%% Solve with P0-method

Zin = xfmr2zin(Z0, ZL, thetaA+thetaB);
Zin_x = xfmr2zin(Z0, ZL, thetaB);

P0 = 1./2.*Vg.^2.*real(Zin)./( (real(Zin) + real(Zg)).^2 + (imag(Zin) + imag(Zg)).^2 ); % From Pozar_4e, page 77, eq. 2,76

thetaX = angle(Zin_x);
Ix_P0based = sqrt( 2.*P0./cos(thetaX)./Zin_x );
Vx_P0based = sqrt(2.*P0.*Zin_x./cos(thetaX));
IL_P0based = sqrt( 2.*P0./cos(0)./ZL );


%% Print reuslts

VI_theta = abs(angle(Vx)-angle(Ix));
Px = 1/2.*abs(Vx).*abs(Ix).*cos(VI_theta);

barprint("ABCD Matrix Based");
displ("IL: ", abs(IL), " A @ ", 180./pi.*angle(IL), " deg");
displ("Vx: ", abs(Vx), " V @ ", 180./pi.*angle(Vx), " deg");
displ("Ix: ", abs(Ix), " A @ ", 180./pi.*angle(Ix), " deg");

displ();
displ("Px = ", Px, " W");

displ()
displ("arg(Ix) - arg(IL): ", 180./pi.*(angle(Ix)-angle(IL)), " deg");
displ("arg(Vx) - arg(Ix): ", 180./pi.*(angle(Vx)-angle(Ix)), " deg");

displ();
barprint("P0 Based");

displ("IL: ", abs(IL_P0based), " A @ ", 180./pi.*angle(IL_P0based), " deg");
displ("Vx: ", abs(Vx_P0based), " V @ ", 180./pi.*angle(Vx_P0based), " deg");
displ("Ix: ", abs(Ix_P0based), " A @ ", 180./pi.*angle(Ix_P0based), " deg");

displ()
displ("P0 = ", P0, " W");

displ()
displ("arg(Ix) - arg(IL): ", 180./pi.*(angle(Ix_P0based)-angle(IL_P0based)), " deg");
displ("arg(Vx) - arg(Ix): ", 180./pi.*(angle(Vx_P0based)-angle(Ix_P0based)), " deg");





