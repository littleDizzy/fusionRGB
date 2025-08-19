function  generateCWT(signal, chooseGate,clutterGate,save_path_CWT)
    % MATLAB 代码：计算 CWT 并进行归一化
    %%%%
    fs = 1000; % 采样频率（Hz）
    signal_target = signal(:, chooseGate); % 第 7 距离门（目标）
    signal_clutter = signal(:, clutterGate); % 杂波距离门（杂波）
    
    % 计算目标门的 CWT
    [cwt_target, f, t] = cwt(signal_target, fs);
    cwt_target = abs(cwt_target); % 取幅值谱
    
    % 计算杂波门的 CWT
    cwt_clutter_all = zeros(size(cwt_target,1), size(cwt_target,2), size(signal_clutter,2));
    for i = 1:size(signal_clutter,2)
        [cwt_tmp, ~, ~] = cwt(signal_clutter(:,i), fs);
        cwt_clutter_all(:,:,i) = abs(cwt_tmp);
    end
    
    % 计算杂波门 CWT 的均值 & 标准差
    mu_clutter = mean(cwt_clutter_all, 3);
    sigma_clutter = std(cwt_clutter_all, 0, 3);
    
    % 计算 NTFD
    ntfd_cwt = (cwt_target - mu_clutter) ./ sigma_clutter;
    
    % 显示 NTFD 结果
    fig = figure('Visible', 'off');
    imagesc(t, f, ntfd_cwt);
    axis xy;
    clim([0,10]);
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('NTFD CWT 时频分布');
    colormap('gray'); % 使用灰度图
    colorbar;
    
    % 捕获图像数据
    frame = getframe(gca);
    img = frame2im(frame);
    
    % 调整图像尺寸至 224×224
    img = imresize(img, [224, 224]);
    % 保存为 PNG 灰度图像
    imwrite(img, save_path_CWT);
    
    % 关闭 figure
    close(fig);
end
