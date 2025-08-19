function signal_seg=slidingWindow(signal,col_select,win_length,step_size)
%%滑动窗口分割数据集
%输入：col_select 选择要切分的距离门，win_length窗口长度，step_size步长
%输出：复数序列signal_seg
%by wh

% signal=abs(complexSeq17HH);
%signal=real(complexSeq17HH);%%用实部！用实部！用实部！
% col_select = [1,9];%按顺序选择一个杂波门和目标门
% win_length = 512;     % 窗口长度
% step_size = 512;        % 步长
[num_samples, ~] = size(signal);  % 获取矩阵的维度
num_windows = floor((num_samples - win_length) / step_size) + 1;  % 计算窗口数量
num_signals = length(col_select);  % 选择的信号数量
% 初始化存储窗口数据的三维矩阵
signal_seg = zeros(win_length, num_windows, num_signals);

% 滑动窗口操作
for w = 1:num_windows
    start_idx = (w - 1) * step_size + 1;  % 当前窗口的起始索引
    end_idx = start_idx + win_length - 1; % 当前窗口的结束索引
    for c = 1:num_signals  % 针对特定列信号进行操作
        signal_seg(:, w, c) = signal(start_idx:end_idx, col_select(c));  % 提取窗口数据
    end
end
end