function generateSTFT(signal, save_path_STFT)

    % 输入参数:
    % signal: 输入信号
    % outputFileName: 输出文件名 (包括路径和扩展名)
    
    % 设置 stft 参数
    fs=1000;%Hz
    windowLength = 128; % 窗口长度
    overlap = windowLength / 2; % 窗口重叠量
    nfft = 1024; % FFT 点数
    % 计算 STFT
    [S, F, T] = stft(signal, fs, ...
        'Window', hamming(windowLength), ...
        'OverlapLength', overlap, ...
        'FFTLength', nfft);
    %%%
    % 计算幅度（转换为 dB）
    S_magnitude = abs(S); % 取绝对值
    S_magnitude_db = 20*log10(S_magnitude); % 转换为 dB

    % 绘制 STFT 图
    figure;
    imagesc(T, F, S_magnitude_db);
    %colorbar;
    colormap('jet'); % 设置颜色映射
    clim([0, 70]); % 小于0映射为第一个颜色，大于0映射为最后一个颜色
    axis xy; % 确保频率轴方向正确
    %%%
    % 去除坐标轴和标签
    set(gca, 'XTick', []); % 移除 x 轴的刻度
    set(gca, 'YTick', []); % 移除 y 轴的刻度
    set(gca, 'XColor', 'none'); % 移除 x 轴的颜色
    set(gca, 'YColor', 'none'); % 移除 y 轴的颜色
    
    % 选择颜色映射    
    % 保存为 PNG 图像
    saveas(gcf, save_path_STFT); % 保存为 PNG 格式
    close(gcf); % 关闭图像窗口
end
