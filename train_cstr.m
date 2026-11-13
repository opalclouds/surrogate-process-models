%% CSTR surrogate — train and benchmark
clear; clc; rng(0);

data = readtable('cstr_data.csv');
% --- Clean failed solves ----------------------------------------------
bad = any(isnan([data.Ca_ss, data.T_ss, data.conv]), 2) | ...
    any(isinf([data.Ca_ss, data.T_ss, data.conv]), 2);
fprintf('Removing %d rows with NaN/Inf (%.1f%% of data)\n', ...
    sum(bad), 100*sum(bad)/height(data));
data(bad,:) = [];
fprintf('Remaining: %d rows\n', height(data));

X = [data.q, data.Caf, data.Tf, data.Tc, data.V, data.UA];
Y = [data.Ca_ss, data.T_ss, data.conv];

N = height(data);
idx = randperm(N);
nTr = round(0.8*N);
Xtr = X(idx(1:nTr),:);     Ytr = Y(idx(1:nTr),:);
Xte = X(idx(nTr+1:end),:); Yte = Y(idx(nTr+1:end),:);

mx = mean(Xtr); sx = std(Xtr);
my = mean(Ytr); sy = std(Ytr);
Xtr_s = (Xtr - mx)./sx;  Xte_s = (Xte - mx)./sx;
Ytr_s = (Ytr - my)./sy;

layers = [
    featureInputLayer(6)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(3)
    regressionLayer];

opts = trainingOptions('adam', ...
    'MaxEpochs', 400, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',150, ...
    'Shuffle','every-epoch', ...
    'Verbose', true, 'VerboseFrequency', 100, ...
    'Plots','none');

net = trainNetwork(Xtr_s, Ytr_s, layers, opts);

Pred = predict(net, Xte_s) .* sy + my;

names = {'Ca_ss (mol/L)', 'T_ss (K)', 'Conversion'};
fprintf('\n--- Accuracy on held-out test set ---\n');
for i = 1:3
    r2  = 1 - sum((Yte(:,i)-Pred(:,i)).^2)/sum((Yte(:,i)-mean(Yte(:,i))).^2);
    mae = mean(abs(Yte(:,i)-Pred(:,i)));
    fprintf('%-16s R2 = %.4f   MAE = %.4f\n', names{i}, r2, mae);
end

% --- Speed: ODE solve vs surrogate -------------------------------------
k0=7.2e10; E_R=8750; dH=-5e4; rho=1000; Cp=0.239;
nB = min(100, size(Xte,1));       % 100 points is enough; ODE is slow

tic
for i = 1:nB
    f = @(t,y) cstr(t,y, Xte(i,1),Xte(i,2),Xte(i,3),Xte(i,4),Xte(i,5),Xte(i,6), ...
        k0,E_R,dH,rho,Cp);
    [~,~] = ode45(f, [0 30], [Xte(i,2); Xte(i,3)]);
end
t_ode = toc/nB;

tic
for rep = 1:100
    predict(net, Xte_s);
end
t_nn = toc/(100*size(Xte,1));

fprintf('\n--- Speed ---\n');
fprintf('ODE solver : %9.3f ms/point\n', t_ode*1e3);
fprintf('Surrogate  : %9.4f ms/point\n', t_nn*1e3);
fprintf('Speedup    : %9.0fx\n', t_ode/t_nn);

% --- Parity plots ------------------------------------------------------
figure('Position',[100 100 1200 350]);
for i = 1:3
    subplot(1,3,i)
    scatter(Yte(:,i), Pred(:,i), 8, 'filled', 'MarkerFaceAlpha',0.3); hold on
    lims = [min(Yte(:,i)) max(Yte(:,i))];
    plot(lims, lims, 'r--', 'LineWidth',1.2)
    xlabel('Simulated'); ylabel('Predicted'); title(names{i})
    axis square; grid on
end
saveas(gcf,'cstr_parity.png');

save('cstr_surrogate.mat','net','mx','sx','my','sy');
fprintf('\nSaved cstr_parity.png and cstr_surrogate.mat\n');

function dydt = cstr(~, y, q, Caf, Tf, Tc, V, UA, k0, E_R, dH, rho, Cp)
Ca = y(1); T = y(2);
rA = k0*exp(-E_R/T)*Ca;
dydt = [ q/V*(Caf-Ca) - rA;
    q/V*(Tf-T) + (-dH)/(rho*Cp)*rA + UA/(V*rho*Cp)*(Tc-T) ];
end