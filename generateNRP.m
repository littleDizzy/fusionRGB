function generateNRP(signal, chooseGate,clutterGate,save_path_RP)
%输入signal为n*1的信号，输出该信号的递归图
    % 参数设置
    embedding_dimension = 3; % 嵌入维度
    embedding_delay = 11;     % 嵌入时延
    signal_target = signal(:, chooseGate); % 第 7 距离门（目标）
    signal_clutter = signal(:, clutterGate); % 杂波距离门（杂波）

    % 计算目标门的递归矩阵
    recursion_matrix = compute_recursion_matrix(signal_target, embedding_dimension, embedding_delay);
    
    
    % 计算杂波门的递归矩阵
    RP_clutter_all = zeros(size(recursion_matrix,1), size(recursion_matrix,2), size(signal_clutter,2));
    for i = 1:size(signal_clutter,2)
        recursion_matrix_tmp = compute_recursion_matrix(signal_clutter(:,i), embedding_dimension, embedding_delay);
        RP_clutter_all(:,:,i) = abs(recursion_matrix_tmp);
    end
    
        % 计算杂波递归图的统计特性
    mu_clutter = median(RP_clutter_all, 3);       % 中值矩阵（抗离群点）
    sigma_clutter = std(RP_clutter_all, 0, 3);    % 标准差矩阵
    max_clutter = max(RP_clutter_all, [], 3);     % 最大值矩阵
    
    % 自适应阈值公式（可调整权重）
    alpha = 0;  % 控制阈值严格程度（通常2~5）
    threshold_map = mu_clutter + alpha * sigma_clutter;

    % 目标二值化
    binary_target = recursion_matrix < threshold_map;
    
    figure;
    %fig = figure('Visible', 'off');
    imshow(~binary_target);  % 强制按二值图像显示
    colormap(gray(2));  % 确保颜色映射为黑白
    colorbar;
    title('Gray Normalized Recurrence Plot');
    xlabel('Time');
    ylabel('Time');
    % 捕获图像数据
    frame = getframe(gca);
    img = frame2im(frame);

    % 保存为 PNG 图像
    %imwrite(img, save_path_RP);

    % 关闭 figure
    %close(fig);
end


% 递归矩阵计算函数
function D = compute_recursion_matrix(data, embedding_dimension, embedding_delay)
    % 将时间序列嵌入到高维空间
    embedded_data = embed_time_series(data, embedding_dimension, embedding_delay);
    threshold=5;
    % 计算递归矩阵
    N = size(embedded_data, 1);
    D = zeros(N);
    for i = 1:N
        for j = 1:N
            % 计算两个点之间的欧氏距离
            % dist= norm(embedded_data(i, :) - embedded_data(j, :));
            % D(i,j) = (dist >= threshold); % 二值化（1=递归点，0=非递归点）
            D(i,j) = norm(embedded_data(i, :) - embedded_data(j, :));
        end
    end
end



% 时间序列嵌入函数
%embedded_data.shape=980*5. 嵌入向量的长度1024-(m-1)*t=980，维度为m=5.
function embedded_data = embed_time_series(data, embedding_dimension, embedding_delay)
    N = length(data);
    embedded_data = zeros(N - (embedding_dimension - 1) * embedding_delay, embedding_dimension);
    for i = 1:embedding_dimension
        start_idx = (i - 1) * embedding_delay + 1;
        end_idx = N - (embedding_dimension - i) * embedding_delay;
        embedded_data(:, i) = data(start_idx:end_idx);
    end
end