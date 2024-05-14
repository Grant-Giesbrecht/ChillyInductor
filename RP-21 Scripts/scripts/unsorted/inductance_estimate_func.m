f = 10e9;

L_ = 1e-6/100;
C_ = 129e-12/100;
len = 0.5;
N = 10;

termZ = 88;

Z0 = sqrt(L_/C_);

displ("Function ",N, " Stages:");
displ("  Z0 = ",Z0, " Ohms");

% [Ztot, L, Zta] = sim_TLIN(L_, C_, len, N, f);

% figure(1);
% hold off;
% plot(imag(Zta));
% % hold on;
% % plot(real(Zta));

displ("Z(N=", N, ") = ", Ztot);
displ("L(N=", N, ") = ", L*1e9, " nH"); 

Ztot_all = ones(1, N);

N_list = 160:170;
Ls = zeros(1, numel(N_list));
Zs = zeros(1, numel(N_list));
idx = 0;
for nn = N_list
	idx = idx + 1;
	[Z,L,~] = sim_TLIN(L_, C_, len, nn, f, termZ);
	Ls(idx) = L;
	Zs(idx) = Z;
end

figure(2);
hold off;
plot(N_list, real(Zs), 'Color', [0, 0, .7]);
hold on;
plot(N_list, imag(Zs), 'Color', [0.7, 0, 0]);
plot(N_list, abs(Zs), 'Color', [0, .7, 0]);
ylim([0, 100]);
grid on;
xlabel("Number of Stages");
ylabel("Impedance (Ohms)");
legend("Re", "Im", "Abs");

figure(3);
hold off;
plot(N_list, Ls.*1e9);
grid on
ylabel("L Estimate (nH)");
xlabel("Number of Stages");
% ylim([0, 10]);

idx_150 = (N_list <= 150);

figure(4);
hold off;
plotsc(Zs(idx_150), 'Domain', 'Z', 'LineWidth', 3);

function [Ztl, Lequiv, Ztot_all] = sim_TLIN(L_, C_, len, N, f, term_Z)

	L_ = L_*1e6;
	C_ = C_*1e6;

	L_tot = L_*len;
	C_tot = C_*len;

	ZL_ = Zl(L_tot/N, f);
	ZC_ = Zc(C_tot/N, f);
	
% 	displ("  ZL_ = ", ZL_ );
% 	displ("  ZC_ = ", ZC_ );
% 	displ("  N = ", N );

	idx = 1;
	Ztot = term_Z; % C_ is shorted for last segment
	Ztot_all(idx) = Ztot;
	
	for n = 1:N
		idx = idx + 1;
% 		displ("    Ztot = ", Ztot);
		
		Ztot = pall(Ztot, ZC_);
% 		Ztot = Ztot.*ZC_./(Ztot + ZC_);
		Ztot = Ztot + ZL_;

		
		Ztot_all(idx) = Ztot;
		
	end

	Ztot = Ztot./1e6;
	
	X = imag(Ztot);
	Lequiv = X/2/pi/f;
	
	Ztl = Ztot;
	
end