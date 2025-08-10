function  generateSPWVD(signal, chooseGate,clutterGate,save_path_SPWVD)
    %功能：将1024*14的信号，生成1张SPWVD（NTFD）图
    % 输入参数:
    % signal: 输入信号1024*14,
    % chooseGate：要检测的距离门,
    % clutterGate杂波距离门，如clutterGate=[1:5,10:14]
    % outputFileName: 输出文件名 (包括路径和扩展名)
    fs=1000;%Hz
    signal_target = signal(:,chooseGate); % 第 7 距离门（目标）
    signal_clutter = signal(:,clutterGate); % 杂波距离门（杂波）
    
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
    %figure;
    fig = figure('Visible', 'off');
    imagesc(t, f, ntfd_tf);
    axis xy;
    clim([0,10]);%使 0 或更小的值映射到颜色图中的第一种颜色，100 或更大的值映射到颜色图中的最后一种颜色
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('NTFD 时频分布');
    %colormap('jet');
    colormap('gray'); % 使用灰度图
    colorbar;
    % 去除坐标轴和标签
    % 捕获图像数据
    frame = getframe(gca);
    img = frame2im(frame);
        % 调整图像尺寸至 224×224
    img = imresize(img, [224, 224]);
    % 保存为 PNG 图像
    imwrite(img, save_path_SPWVD);
    
    % 关闭 figure
    close(fig);
end