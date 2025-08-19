function generateRP(signal,save_path_RP)
%输入signal为n*1的信号，输出该信号的递归图
    % 参数设置
    embedding_dimension = 3; % 嵌入维度
    embedding_delay = 11;     % 嵌入时延
    
    % 计算递归矩阵
    recursion_matrix = compute_recursion_matrix(signal, embedding_dimension, embedding_delay);
    
    % 灰度归一化处理
    min_val = min(recursion_matrix(:)); % 递归矩阵的最小值
    max_val = max(recursion_matrix(:)); % 递归矩阵的最大值
    gray_image = (recursion_matrix - min_val) / (max_val - min_val) * 255; % 归一化到 [0, 255]
    % 将归一化后的递归图缩放到224×224
    resized_matrix = imresize(gray_image, [224,224]);
    % 显示灰度图像
    % figure;
    fig = figure('Visible', 'off');
    imshow(resized_matrix, [0, 255]);
    colormap gray;
    title('Gray Normalized Recurrence Plot');
    xlabel('Time');
    ylabel('Time');
    % 捕获图像数据
    frame = getframe(gca);
    img = frame2im(frame);

    % 保存为 PNG 图像
    imwrite(img, save_path_RP);

    % 关闭 figure
    close(fig);
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