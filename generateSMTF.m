function SMTF = generateSMTF(timeSeries, numQuantiles,save_path_SMTF)
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

    %  绘制并保存 SMTF
    fig = figure('Visible', 'off'); % 不显示图形窗口
    imagesc(SMTF);
    colormap('jet');
    % colorbar;
    % title('SMTF');
    % xlabel('Time Index');
    % ylabel('Time Index');
    % 去除坐标轴和标签
    set(gca, 'XTick', []); % 移除 x 轴的刻度
    set(gca, 'YTick', []); % 移除 y 轴的刻度
    set(gca, 'XColor', 'none'); % 移除 x 轴的颜色
    set(gca, 'YColor', 'none'); % 移除 y 轴的颜色
    saveas(gcf,save_path_SMTF);
    close(fig);  % 关闭图形窗口
end
