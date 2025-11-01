clear;close;clc;
%% ======================= 配置与依赖检查 ==============================
assert(exist('trainNetwork','file')==2, '需要 Deep Learning Toolbox (trainNetwork)。');

rng(2025);                                % 固定随机种子
c = 299792458;                          % 光速 (m/s)

%% 场景与采样
areaSize = [60, 40];                    % 场地 [X, Y] (m)
TRP = [ 0,          0;                  % 4 个TRP
        areaSize(1),0;
        areaSize(1),areaSize(2);
        0,          areaSize(2) ];
M = size(TRP,1);

Nscatter   = 20;                        % 散射体数量
SC = [ areaSize(1)*rand(Nscatter,1), areaSize(2)*rand(Nscatter,1) ];

% 频域“探针”参数（只需要频率点，不用真的发波形）
Nsc = 32;                               % 子载波数（也是图像的行数）
df  = 3e6;                             % 子载波间隔 (Hz)
f   = (-Nsc/2:Nsc/2-1).' * df;          % 基带频率向量 (Nsc×1)

% 数据集规模（可适当调小以加速）
Ntrain = 3000; Nval = 500; Ntest = 200;
Ntot   = Ntrain + Nval + Ntest;

SNRdB = 25;                             % CSI 的等效 SNR
SNRlin = 10.^(SNRdB/10);
losProb    = 0.6;                       % 链路为 LoS 的概率
reflAtten  = 0.6;                       % 反射额外衰减（幅度）
extraPaths = 2;                         % 额外弱多径条数（增强“指纹”）
usePhaseJitter = false;                 % 是否加整体相位抖动（默认为否）

%% 生成数据：X (32×32×(2*M)×N), Y (N×2)  —— 复数CSI→图像
H = 32; W = 32; C = 2*M;                % 图像尺寸
X = zeros(H, W, C, Ntot, 'single');
Y = zeros(Ntot, 2, 'single');           % [x, y]（米）

for n = 1:Ntot
    % 随机 UE 位置（避开边界 2m）
    UE = [2 + (areaSize(1)-4)*rand, 2 + (areaSize(2)-4)*rand];
    Y(n,:) = single(UE); % 直接学习真实坐标（不归一化也可行）；若想归一化，请注释此行并用下一行：
    % Y(n,:) = single([UE(1)/areaSize(1), UE(2)/areaSize(2)]);  % 若启用则同时修改训练后反归一化部分

    chanFeat = zeros(H, W, C, 'single');
    ch = 0;

    for m = 1:M
        % 判定 LoS / NLoS
        isLoS = rand < losProb;

        % 主路径（LoS 或 单次反射）
        if isLoS
            d_main = hypot(UE(1)-TRP(m,1), UE(2)-TRP(m,2));
            tau_main = d_main / c;
            amp_main = 1 / max(d_main,1);
        else
            % 选总路长最短的散射体：|UE-S|+|S-TRP_m|
            rUE = hypot(SC(:,1)-UE(1),  SC(:,2)-UE(2));
            rTR = hypot(SC(:,1)-TRP(m,1), SC(:,2)-TRP(m,2));
            [d_bounce, idxS] = min(rUE + rTR);
            d_bounce = max(d_bounce, 1e-3) + 1e-6;  % 防等号/0
            tau_main = d_bounce / c;
            amp_main = reflAtten / max(d_bounce,1);
        end

        % 额外弱多径（从随机散射体生成）
        taus = tau_main; amps = amp_main;
        for l = 1:extraPaths
            k = randi(Nscatter);
            d_ex = hypot(UE(1)-SC(k,1), UE(2)-SC(k,2)) + hypot(SC(k,1)-TRP(m,1), SC(k,2)-TRP(m,2));
            d_ex = max(d_ex, 1e-3);
            taus(end+1,1) = d_ex / c; 
            amps(end+1,1) = (reflAtten^2) / d_ex; 
        end

        % 可选整体相位抖动（模拟未知参考相位）
        if usePhaseJitter
            phi0 = 2*pi*rand;
        else
            phi0 = 0;
        end

        % 合成 CSI：H(f) = Σ a_l * exp(-j*2π f τ_l)
        Hf = zeros(Nsc,1);
        for l = 1:numel(taus)
            Hf = Hf + amps(l) * exp(-1j*2*pi*f*taus(l));
        end
        Hf = Hf * exp(1j*phi0);

        % 加 AWGN 噪声（按每TRP CSI功率设定）
        sigPow = mean(abs(Hf).^2);
        noise  = (randn(size(Hf))+1j*randn(size(Hf))) * sqrt(sigPow/(2*SNRlin));
        Hf_noisy = Hf + noise;

        % —— CSI→图像通道：实部/虚部，各复制成 32×32 —— 
        reImg = repmat(real(Hf_noisy), 1, W);  % 32×32
        imImg = repmat(imag(Hf_noisy), 1, W);  % 32×32

        ch = ch + 1; chanFeat(:,:,ch) = single(reImg);
        ch = ch + 1; chanFeat(:,:,ch) = single(imImg);
    end

    % 每样本做 z-score 归一化（跨所有像素与通道）
    mu = mean(chanFeat(:)); sd = std(chanFeat(:))+1e-8;
    chanFeat = (chanFeat - mu)/sd;

    X(:,:,:,n) = chanFeat;
end

%% 划分数据集（打乱）
idx = randperm(Ntot);
idxTrain = idx(1:Ntrain);
idxVal   = idx(Ntrain+1:Ntrain+Nval);
idxTest  = idx(Ntrain+Nval+1:end);

XTrain = X(:,:,:,idxTrain); YTrain = Y(idxTrain,:);
XVal   = X(:,:,:,idxVal);   YVal   = Y(idxVal,:);
XTest  = X(:,:,:,idxTest);  YTest  = Y(idxTest,:);

%% 搭建轻量 CNN 回归网络
inputSize = [H W C];

layers = [
    imageInputLayer(inputSize, "Normalization","none","Name","in")

    convolution2dLayer(3, 32, "Padding","same","Name","conv1")
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")

    convolution2dLayer(3, 32, "Padding","same","Name","conv2")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")

    convolution2dLayer(3, 64, "Padding","same","Stride",2,"Name","conv3")  % 下采样到 16x16
    batchNormalizationLayer("Name","bn3")
    reluLayer("Name","relu3")

    convolution2dLayer(3, 64, "Padding","same","Name","conv4")
    batchNormalizationLayer("Name","bn4")
    reluLayer("Name","relu4")

    globalAveragePooling2dLayer("Name","gap")
    fullyConnectedLayer(64,"Name","fc1")
    reluLayer("Name","relu5")
    dropoutLayer(0.1,"Name","drop1")
    fullyConnectedLayer(2,"Name","fc_out")      % 输出 [x,y]（米）
    regressionLayer("Name","reg")];

%% 训练选项
valFreq = max(1, floor(numel(idxTrain)/128));
opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',12, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal, YVal}, ...
    'ValidationFrequency',valFreq, ...
    'Plots','none', ...
    'Verbose',false);

%% 训练
[net, info] = trainNetwork(XTrain, YTrain, layers, opts);

%% 推断与评估（在测试集）
YPred = predict(net, XTest, 'MiniBatchSize',256);

% 若上面把 Y 归一化到了 [0,1]，这里要反归一化：
% YPred(:,1) = YPred(:,1)*areaSize(1);
% YPred(:,2) = YPred(:,2)*areaSize(2);
% YTest(:,1) = YTest(:,1)*areaSize(1);
% YTest(:,2) = YTest(:,2)*areaSize(2);

err = sqrt(sum((YPred - YTest).^2, 2));         % 欧式误差（米）
[es,ord] = sort(err); cdfy = (1:numel(es)).'/numel(es);
P10 = es(max(1,ceil(0.10*numel(es))));
P50 = es(max(1,ceil(0.50*numel(es))));
P90 = es(max(1,ceil(0.90*numel(es))));

fprintf('=== AI直定位（CNN回归）测试结果 ===\n');
fprintf('测试样本: %d\n', numel(err));
fprintf('误差: 均值=%.3f m, 中位=%.3f m, P10=%.3f m, P90=%.3f m, 最大=%.3f m\n', ...
    mean(err), median(err), P10, P90, max(err));

%% ======================= 可视化 1：训练曲线 =========================
figure('Name','Training Curves'); grid on; hold on;
plot(info.TrainingLoss,'-','LineWidth',1.5); 
if isfield(info,'ValidationLoss') && ~all(isnan(info.ValidationLoss))
    plot(info.ValidationLoss,'--','LineWidth',1.5);
end
xlabel('Iteration'); ylabel('Loss (MSE)');
legend('Train','Val','Location','best'); title('Training / Validation Loss');

%% ======================= 可视化 2：几何散点 + 误差向量 ===============
figure('Name','Test Geometry (True vs Pred)'); clf; hold on; axis equal; grid on;
rectangle('Position',[0,0,areaSize],'EdgeColor',[0.6 0.6 0.6]);
plot(TRP(:,1), TRP(:,2), 'ks', 'MarkerSize',8, 'LineWidth',1.2);
plot(YTest(:,1), YTest(:,2), 'g.', 'MarkerSize',12);           % 真值
plot(YPred(:,1), YPred(:,2), 'rx', 'MarkerSize',6, 'LineWidth',1); % 估计
% 误差向量（抽样一部分，避免拥挤）
nShow = min(400, size(YTest,1));
idv   = randperm(size(YTest,1), nShow);
quiver(YTest(idv,1), YTest(idv,2), YPred(idv,1)-YTest(idv,1), YPred(idv,2)-YTest(idv,2), ...
       0, 'Color',[0.85 0.2 0.2], 'LineWidth',0.8);
legend('TRP','GT','Pred','ErrVec','Location','bestoutside');
title(sprintf('AI直定位：Test N=%d, Err median=%.2f m', numel(err), median(err)));
xlabel('X (m)'); ylabel('Y (m)');

%% ======================= 可视化 3：误差直方图 ========================
figure('Name','Error Histogram'); grid on;
histogram(err, max(10,round(sqrt(numel(err)))), 'Normalization','pdf');
xlabel('定位误差 (m)'); ylabel('概率密度'); title('测试集误差直方图');

%% ======================= 可视化 4：误差 CDF ==========================
figure('Name','Error CDF'); grid on; hold on;
plot(es, cdfy, 'LineWidth',1.5);
yl = ylim;
plot([P10 P10], yl, 'k--');
plot([P50 P50], yl, 'k--');
plot([P90 P90], yl, 'k--');
text(P10, 0.05, sprintf('P10=%.2f m',P10), 'HorizontalAlignment','left','VerticalAlignment','bottom');
text(P50, 0.35, sprintf('P50=%.2f m',P50), 'HorizontalAlignment','left','VerticalAlignment','bottom');
text(P90, 0.75, sprintf('P90=%.2f m',P90), 'HorizontalAlignment','left','VerticalAlignment','bottom');
xlabel('定位误差 (m)'); ylabel('概率'); title('测试集误差 CDF');

%% ======================= 可视化 5：示例CSI通道图 =====================
% 展示一个测试样本的若干通道（某个TRP的实/虚部）
s = ord( max(1, round(0.5*numel(ord))) );   % 选一个中位误差样本
img = XTest(:,:,:,s);
figure('Name','Example CSI Channels'); colormap parula;
for m = 1:M
    subplot(2,M,m);    imagesc(img(:,:,2*m-1)); axis image off; title(sprintf('TRP%d-Re',m));
    subplot(2,M,M+m);  imagesc(img(:,:,2*m  )); axis image off; title(sprintf('TRP%d-Im',m));
end
sgtitle('样本的 CSI 通道图（归一化后）');
