% This gives the correct answer for Zin!

Zterm = 100; % Ohm
L_ = 1e-6; % H/m
C_ = 121e-12; % F/m


N = 150; % Num stages
phi_deg = 90; % Line length (deg)
f = 10e9;% Frequency (Hz)

Vp = 1/sqrt(L_*C_);
lambda = Vp/f;
len = lambda*phi_deg/360;
dx = len/N;
Cx = C_*dx;
Lx = L_*dx;

% net = [1, 0; 1/Zterm, 1]; % Initilize network with termination
net = [1, 0; 0, 1]; % Initialize with identity matrix

halfcap = [1, 0; 1/Zc(Cx/2, f), 1];
ind = [1, Zl(Lx, f); 0, 1];

for n = 1:N
	
	net = net * halfcap * ind * halfcap;
	
end

% Estimate Zin
Zin = (Zterm*net(1, 1) + net(1, 2) )/(Zterm*net(2, 1) + net(2, 2));

% Convert to approx characteristic impedance
Z0_est = sqrt(Zin*Zterm);

displ("Estimated Z0 for:");
displ("  L' = ", L_*1e6, " uH/m");
displ("  C' = ", C_*1e12, " pF/m");
displ("  N  = ", N, " stages");
displ();
displ("  Zin= ", Zin, " Ohms");
displ("  Z0 = ", real(Z0_est), " Ohms");













