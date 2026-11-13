%% Heat exchanger surrogate — train and benchmark
clear; clc; rng(0);

data = readtable('heat_exchanger_data.csv');

X = [data.m_hot, data.m_cold, data.T_hot, data.T_cold, data.A, data.U];
Y = [data.Q, data.T_hot_o, data.T_cold_o];

% --- Split -------------------------------------------------------------
N = height(data);
idx = randperm(N);
nTr = round(0.8*N);
Xtr = X(idx(1:nTr),:);   Ytr = Y(idx(1:nTr),:);
Xte = X(idx(nTr+1:end),:); Yte = Y(idx(nTr+1:end),:);

% --- Scale -------------------------------------------------------------
% U is ~1000 and flows are ~1. Without scaling the large-magnitude inputs
% dominate the gradients and the network cannot learn.
mx = mean(Xtr); sx = std(Xtr);
my = mean(Ytr); sy = std(Ytr);

Xtr_s = (Xtr - mx) ./ sx;   Xte_s = (Xte - mx) ./ sx;
Ytr_s = (Ytr - my) ./ sy;

% --- Network -----------------------------------------------------------
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
    'MaxEpochs', 300, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',100, ...
    'Shuffle','every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots','none');

net = trainNetwork(Xtr_s, Ytr_s, layers, opts);

% --- Evaluate in original units ---------------------------------------
Pred_s = predict(net, Xte_s);
Pred   = Pred_s .* sy + my;

names = {'Q (W)', 'T_hot_out (K)', 'T_cold_out (K)'};
fprintf('\n--- Accuracy on held-out test set ---\n');
for i = 1:3
    r2  = 1 - sum((Yte(:,i)-Pred(:,i)).^2) / sum((Yte(:,i)-mean(Yte(:,i))).^2);
    mae = mean(abs(Yte(:,i)-Pred(:,i)));
    fprintf('%-16s R2 = %.4f   MAE = %.4f\n', names{i}, r2, mae);
end

% --- Speed comparison --------------------------------------------------
cp_hot = 2200; cp_cold = 4180;
nB = size(Xte,1);

tic
for rep = 1:10
    for i = 1:nB
        Cmin = min(Xte(i,1)*cp_hot, Xte(i,2)*cp_cold);
        Cmax = max(Xte(i,1)*cp_hot, Xte(i,2)*cp_cold);
        Cr   = Cmin/Cmax;
        NTU  = Xte(i,6)*Xte(i,5)/Cmin;
        if abs(Cr-1) < 1e-6
            e = NTU/(1+NTU);
        else
            e = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr)));
        end
        q = e*Cmin*(Xte(i,3)-Xte(i,4));
    end
end
t_sim = toc/(10*nB);

tic
for rep = 1:10
    predict(net, Xte_s);
end
t_nn = toc/(10*nB);

fprintf('\n--- Speed ---\n');
fprintf('Physics model : %8.3f us/point\n', t_sim*1e6);
fprintf('Surrogate     : %8.3f us/point\n', t_nn*1e6);
fprintf('Speedup       : %8.1fx\n', t_sim/t_nn);

% --- Parity plot -------------------------------------------------------
figure('Position',[100 100 1200 350]);
for i = 1:3
    subplot(1,3,i)
    scatter(Yte(:,i), Pred(:,i), 8, 'filled', 'MarkerFaceAlpha', 0.3)
    hold on
    lims = [min(Yte(:,i)) max(Yte(:,i))];
    plot(lims, lims, 'r--', 'LineWidth', 1.2)
    xlabel('Simulated'); ylabel('Predicted'); title(names{i})
    axis square; grid on
end
saveas(gcf, 'parity_plots.png');
fprintf('\nSaved parity_plots.png\n');

save('surrogate_model.mat','net','mx','sx','my','sy');