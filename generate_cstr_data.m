%% Dynamic CSTR — data generation (expensive physics)
% Non-isothermal CSTR with exothermic first-order reaction A -> B.
% Each point requires an ODE solve, so the physics is genuinely costly.

clear; clc; rng(0);

N = 2000;

% --- Operating envelope ------------------------------------------------
q      = 50   + 150*rand(N,1);     % flow rate, L/min        [50, 200]
Caf    = 0.5  + 1.5*rand(N,1);     % feed concentration, mol/L
Tf     = 300  +  50*rand(N,1);     % feed temperature, K
Tc     = 280  +  60*rand(N,1);     % coolant temperature, K
V      = 80   +  40*rand(N,1);     % reactor volume, L
UA     = 3e4  +  7e4*rand(N,1);    % heat transfer term, J/min K

% --- Fixed kinetics and properties -------------------------------------
k0    = 7.2e10;      % 1/min
E_R   = 8750;        % K
dH    = -5e4;        % J/mol  (exothermic)
rho   = 1000;        % g/L
Cp    = 0.239;       % J/g K

Ca_ss = zeros(N,1); T_ss = zeros(N,1); conv = zeros(N,1);

tic
for i = 1:N
    odefun = @(t,y) cstr(t, y, q(i), Caf(i), Tf(i), Tc(i), V(i), UA(i), ...
        k0, E_R, dH, rho, Cp);
    y0 = [Caf(i); Tf(i)];
    [~, Y] = ode45(odefun, [0 30], y0);      % integrate to steady state
    Ca_ss(i) = Y(end,1);
    T_ss(i)  = Y(end,2);
    conv(i)  = (Caf(i) - Ca_ss(i)) / Caf(i);
end
t_sim = toc;

fprintf('Simulated %d points in %.2f s  (%.3f ms per point)\n', ...
    N, t_sim, 1000*t_sim/N);

T = table(q, Caf, Tf, Tc, V, UA, Ca_ss, T_ss, conv);
writetable(T, 'cstr_data.csv');
fprintf('Wrote cstr_data.csv\n');
fprintf('Conversion: min %.3f  max %.3f\n', min(conv), max(conv));
fprintf('T_ss (K):   min %.1f  max %.1f\n', min(T_ss), max(T_ss));

% --- ODE right-hand side -----------------------------------------------
function dydt = cstr(~, y, q, Caf, Tf, Tc, V, UA, k0, E_R, dH, rho, Cp)
Ca = y(1); T = y(2);
rA = k0 * exp(-E_R/T) * Ca;
dCadt = q/V*(Caf - Ca) - rA;
dTdt  = q/V*(Tf - T) + (-dH)/(rho*Cp)*rA + UA/(V*rho*Cp)*(Tc - T);
dydt  = [dCadt; dTdt];
end