function generateGAF(X,save_path_GASF,save_path_GADF)
    %输入信号和地址，得到GASF和GADF，并分别保存在指定位置
    % (1) 归一化到 [-1, 1]
    X_tilde = (X-max(X)+X-min(X))/(max(X)-min(X));
    % (2) 计算 GASF 和 GADF
    % GASF = X_tilde' * X_tilde - sqrt(1 - X_tilde^2)' * sqrt(1 - X_tilde^2)
    % GADF = sqrt(1 - X_tilde^2)' * X_tilde + X_tilde' * sqrt(1 - X_tilde^2)
    X_tilde = X_tilde(:); % 确保 X_tilde 是列向量
    sqrt_term = sqrt(1 - X_tilde.^2); % 计算 sqrt(1 - X_tilde^2)
    
    GASF = X_tilde * X_tilde' - sqrt_term * sqrt_term'; % Gramian Angular Summation Field 利用和角关系（原文）
    GADF = sqrt_term * X_tilde' + X_tilde * sqrt_term'; % Gramian Angular Difference Field 利用差角关系
    % (3) 绘制并保存 GASF
    fig1 = figure('Visible', 'off'); % 不显示图形窗口
    imagesc(GASF);
    colormap('jet');
    % colorbar;
    % title('Gramian Angular Summation Field (GASF)');
    % xlabel('Time Index');
    % ylabel('Time Index');
    % 去除坐标轴和标签
    set(gca, 'XTick', []); % 移除 x 轴的刻度
    set(gca, 'YTick', []); % 移除 y 轴的刻度
    set(gca, 'XColor', 'none'); % 移除 x 轴的颜色
    set(gca, 'YColor', 'none'); % 移除 y 轴的颜色
    saveas(gcf, save_path_GASF);
    close(fig1);  % 关闭图形窗口

    % (4) 绘制并保存 GADF
    fig2 = figure('Visible', 'off'); % 不显示图形窗口
    imagesc(GADF);
    colormap('jet');
    % colorbar;
    % title('Gramian Angular Difference Field (GADF)');
    % xlabel('Time Index');
    % ylabel('Time Index');
    % 去除坐标轴和标签
    set(gca, 'XTick', []); % 移除 x 轴的刻度
    set(gca, 'YTick', []); % 移除 y 轴的刻度
    set(gca, 'XColor', 'none'); % 移除 x 轴的颜色
    set(gca, 'YColor', 'none'); % 移除 y 轴的颜色
    saveas(gcf,save_path_GADF);
    close(fig2);  % 关闭图形窗口
end

