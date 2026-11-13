%% Heat exchanger surrogate — data generation
% Counter-current shell-and-tube exchanger, effectiveness-NTU method.
% Sweeps operating conditions to build a dataset for ML surrogate training.

clear; clc; rng(0);

N = 3000;                          % number of operating points

% --- Sample the operating envelope -----------------------------------
m_hot  = 0.5  + 4.5*rand(N,1);     % hot stream flow, kg/s   [0.5, 5]
m_cold = 0.5  + 4.5*rand(N,1);     % cold stream flow, kg/s  [0.5, 5]
T_hot  = 350  + 100*rand(N,1);     % hot inlet temp, K       [350, 450]
T_cold = 280  +  40*rand(N,1);     % cold inlet temp, K      [280, 320]
A      = 1    +  19*rand(N,1);     % heat transfer area, m^2 [1, 20]
U      = 200  + 800*rand(N,1);     % overall HTC, W/m^2K     [200, 1000]

cp_hot  = 2200;                    % J/kgK  (oil)
cp_cold = 4180;                    % J/kgK  (water)

% --- Effectiveness-NTU ------------------------------------------------
C_hot  = m_hot  .* cp_hot;
C_cold = m_cold .* cp_cold;
C_min  = min(C_hot, C_cold);
C_max  = max(C_hot, C_cold);
Cr     = C_min ./ C_max;

NTU = (U .* A) ./ C_min;

% Counter-current effectiveness
eff = (1 - exp(-NTU.*(1-Cr))) ./ (1 - Cr.*exp(-NTU.*(1-Cr)));
eff(abs(Cr-1) < 1e-6) = NTU(abs(Cr-1) < 1e-6) ./ (1 + NTU(abs(Cr-1) < 1e-6));

Q        = eff .* C_min .* (T_hot - T_cold);    % duty, W
T_hot_o  = T_hot  - Q ./ C_hot;                 % hot outlet, K
T_cold_o = T_cold + Q ./ C_cold;                % cold outlet, K

% --- Time the simulation ---------------------------------------------
tic
for i = 1:N
    Cmin_i = min(m_hot(i)*cp_hot, m_cold(i)*cp_cold);
    Cmax_i = max(m_hot(i)*cp_hot, m_cold(i)*cp_cold);
    Cr_i   = Cmin_i/Cmax_i;
    NTU_i  = U(i)*A(i)/Cmin_i;
    if abs(Cr_i-1) < 1e-6
        e = NTU_i/(1+NTU_i);
    else
        e = (1-exp(-NTU_i*(1-Cr_i)))/(1-Cr_i*exp(-NTU_i*(1-Cr_i)));
    end
    q = e*Cmin_i*(T_hot(i)-T_cold(i));
end
t_sim = toc;
fprintf('Simulation time for %d points: %.4f s (%.2f us per point)\n', ...
    N, t_sim, 1e6*t_sim/N);

% --- Export -----------------------------------------------------------
T = table(m_hot, m_cold, T_hot, T_cold, A, U, Q, T_hot_o, T_cold_o);
writetable(T, 'heat_exchanger_data.csv');
fprintf('Wrote heat_exchanger_data.csv  (%d rows)\n', height(T));

% --- Sanity check ------------------------------------------------------
fprintf('\nEffectiveness: min %.3f  max %.3f\n', min(eff), max(eff));
fprintf('Duty (kW):     min %.1f  max %.1f\n', min(Q)/1000, max(Q)/1000);