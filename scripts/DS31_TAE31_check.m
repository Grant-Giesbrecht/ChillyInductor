%% Configure problem

thetaA = 20;
thetaB = 30;
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

%% PRint reuslts

VI_theta = abs(angle(Vx)-angle(Ix));
Px = 1/2.*abs(Vx).*abs(Ix).*cos(VI_theta);

% displ("IL: ", IL, " A");
% displ("Vx: ", Vx, " V");
% displ("Ix: ", Ix, " A");
% git
% displ();
displ("IL: ", abs(IL), " A @ ", 180./pi.*angle(IL), " deg");
displ("Vx: ", abs(Vx), " V @ ", 180./pi.*angle(Vx), " deg");
displ("Ix: ", abs(Ix), " A @ ", 180./pi.*angle(Ix), " deg");

displ();
displ("Px = ", Px, " W");


