%%
%学习一些时间序列转二维图像的方法

%%
%%参数初始化
signal=abs(complexSeq311HH(1:1024,1));
n = length(signal);
fs=1000;%采样频率Hz

%%
%1.格拉姆角场(Gramian Angular Field, GAF)
% (1) 归一化到 [-1, 1]
X=signal;
X_tilde = (X-max(X)+X-min(X))/(max(X)-min(X));

% (2) 计算 GASF 和 GADF
% GASF = X_tilde' * X_tilde - sqrt(1 - X_tilde^2)' * sqrt(1 - X_tilde^2)
% GADF = sqrt(1 - X_tilde^2)' * X_tilde + X_tilde' * sqrt(1 - X_tilde^2)
X_tilde = X_tilde(:); % 确保 X_tilde 是列向量
sqrt_term = sqrt(1 - X_tilde.^2); % 计算 sqrt(1 - X_tilde^2)

GASF = X_tilde * X_tilde' - sqrt_term * sqrt_term'; % Gramian Angular Summation Field 利用和角关系（原文）
GADF = sqrt_term * X_tilde' + X_tilde * sqrt_term'; % Gramian Angular Difference Field 利用差角关系

% (3) 绘制并保存 GASF
figure (1);
imagesc(GASF);
colormap('jet');
colorbar;
title('Gramian Angular Summation Field (GASF)');
xlabel('Time Index');
ylabel('Time Index');

% (4) 绘制并保存 GADF
figure (2);
imagesc(GADF);
colormap('jet');
colorbar;
title('Gramian Angular Difference Field (GADF)');
xlabel('Time Index');
ylabel('Time Index');

% % (5) 保存图像 (可选)
% save_path_GASF = 'C:\Users\wh\Desktop\time2image\GASF.png';
% save_path_GADF = 'C:\Users\wh\Desktop\time2image\GADF.png';
% saveas(figure(1), save_path_GASF);
% saveas(figure(2),save_path_GADF);

%%
%2.FDMTF
timeSeries=signal;
numQuantiles=4;
SMTF = computeSMTF(timeSeries, numQuantiles);
imagesc(SMTF);
colorbar;
title('Spectral Markov Transition Field (SMTF)');


function SMTF = computeSMTF(timeSeries, numQuantiles)
    % computeSMTF: 计算频域 Markov Transition Field (SMTF)
    % 输入:
    % - timeSeries: 输入的离散时间序列 (1D array)
    % - numQuantiles: 分位数的数量，用于划分状态区间
    % 输出:
    % - SMTF: 生成的频域 Markov Transition Field 矩阵

    % Step 1. Frequency spectrum signal processing
    % Step 1.1: 计算频谱
    N = length(timeSeries);
    freqSpectrum = abs(fft(timeSeries, N));
    
    % Step 1.2: Min-max 归一化
    normSpectrum = (freqSpectrum - min(freqSpectrum)) / (max(freqSpectrum) - min(freqSpectrum));

    % Step 2. Define the state transition interval of the frequency spectrum amplitude
    % Step 2.1: 按振幅升序排列
    sortedSpectrum = sort(normSpectrum);
    
    % Step 2.2: 根据分位数计算状态区间
    quantiles = quantile(sortedSpectrum, linspace(0, 1, numQuantiles + 1));
    
    % Step 3: Construct the SMTF matrix
    % Step 3.1: 将频谱值分配到状态区间
    states = zeros(size(normSpectrum));
    for i = 1:numQuantiles
        states(normSpectrum >= quantiles(i) & normSpectrum < quantiles(i+1)) = i;
    end
    states(normSpectrum == quantiles(end)) = numQuantiles; % 确保最后一个值被包含
    
    % Step 3.2: 构建一阶 MTF 矩阵
    SMTF = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if i <= j
                % 计算状态间的转移关系
                SMTF(i, j) = states(i) * states(j);
            end
        end
    end

    % 对称化处理
    SMTF = SMTF + SMTF';
    diagIndices = logical(eye(size(SMTF)));
    SMTF(diagIndices) = SMTF(diagIndices) / 2; % 对角线元素归一化
end


%%
%3.STFT

% 计算 STFT
[stftData, f, t_stft] = stft(signal, Fs, 'Window', hamming(windowLength), 'OverlapLength', overlap, 'FFTLength', nfft);

% 绘制正负频率的 STFT 图像
figure('Visible', 'off'); % 不显示图像窗口
surf(t_stft, f, abs(stftData), 'EdgeColor', 'none'); % 绘制频谱幅值
view(2); % 设置为俯视图
axis tight;

% 去除坐标轴和标签
set(gca, 'XTick', []); % 移除 x 轴的刻度
set(gca, 'YTick', []); % 移除 y 轴的刻度
set(gca, 'XColor', 'none'); % 移除 x 轴的颜色
set(gca, 'YColor', 'none'); % 移除 y 轴的颜色

colormap('jet'); % 选择颜色映射
clim([0, 70]); % 将幅度坐标固定在 0 到 70 之间
% colorbar; % 显示颜色条

% 保存为 PNG 图像
saveas(gcf, 'stft_no_labels.png'); % 保存为 PNG
close(gcf); % 关闭图像窗口
%%
%4.SPWVD
fs = 1000; % 采样率
% 假设 data 为 1024×14 复数雷达数据
data=complexSeq311HH(1+1024:1024*2,: );
signal_target = data(:,1); % 第 7 距离门（目标）
signal_clutter = data(:,[1:5,10:14]); % 其他距离门（杂波）

% 计算目标门的 SPWVD
[spwvd_target, f, t] = wvd(signal_target, fs,'smoothedPseudo');
spwvd_target = fftshift(spwvd_target, 1); % 调整频率轴

% 计算杂波门的 SPWVD
spwvd_clutter_all = zeros(size(spwvd_target,1), size(spwvd_target,2), size(signal_clutter,2));
for i = 1:size(signal_clutter,2)
    [spwvd_tmp, ~, ~] = wvd(signal_clutter(:,i), fs,'smoothedPseudo');
    spwvd_clutter_all(:,:,i) = fftshift(spwvd_tmp, 1);
end

% 计算杂波门 SPWVD 的均值 & 标准差
mu_clutter = mean(spwvd_clutter_all, 3);
sigma_clutter = std(spwvd_clutter_all, 0, 3);%0：指定使用 N-1 归一化（无偏估计）.std(A, w, dim)

% 计算 NTFD
ntfd_tf = (spwvd_target - mu_clutter) ./ sigma_clutter;

% 显示 NTFD 结果
figure;
imagesc(t, f, ntfd_tf);
axis xy;
clim([0,100]);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('NTFD 时频分布');
colormap('jet');
colorbar;



