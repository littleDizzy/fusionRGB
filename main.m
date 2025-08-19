%%
%本代码旨在将一维信号转二维信号
%代码大纲：
%1.time2image：学习一维信号转二维信号的方法
%2.slidingWindow：滑动窗口分割数据集，得到复数形式切片
%注：使用前用radarEchoData文件，导入数据集
%默认png像素为875*656
%attention：每一种算法的图片的强度颜色colorbar应该映射在同意范围内
%本代码思路：将一维信号通过GAF、MTF、STFT等转成二维图片，图片转为单通道灰度图，将三张灰度图分别使用RGB通道生成彩色图，放入模型预测。
%copyright by wh&GPT
%%
clc;
%输入参数
dataName='data311';
clutter_select = [1:5,10:14];%选择要处理的距离门
target_select = [7];%根据三特征经典论文，只使用primary cell
win_length = 1024;     % 窗口长度
step_size = 1024;        % 步长
%初始化
signal=abs(complexSeq311HH);
signal_raw=(complexSeq311HH);

target_seg=slidingWindow(signal,target_select,win_length,step_size);%实数滑动窗口切片
clutter_seg=slidingWindow(signal,clutter_select,win_length,step_size);%实数滑动窗口切片
signal_num=size(target_seg,2);%每个距离的信号个数

target_seg_raw=slidingWindow(signal_raw,target_select,win_length,step_size);%复数滑动窗口切片
clutter_seg_raw=slidingWindow(signal_raw,clutter_select,win_length,step_size);%复数滑动窗口切片
%%
% GASF 路径
output_dir_GASF_target = sprintf('D:\\time2image\\%s\\GASF\\target\\', dataName); % 目标波的保存路径
output_dir_GASF_clutter = sprintf('D:\\time2image\\%s\\GASF\\clutter\\', dataName); % 海杂波的保存路径

% GADF 路径
output_dir_GADF_target = sprintf('D:\\time2image\\%s\\GADF\\target\\', dataName); % 目标波的保存路径
output_dir_GADF_clutter = sprintf('D:\\time2image\\%s\\GADF\\clutter\\', dataName); % 海杂波的保存路径

% SMTF 路径
output_dir_SMTF_target = sprintf('D:\\time2image\\%s\\SMTF\\target\\', dataName); % 目标波的保存路径
output_dir_SMTF_clutter = sprintf('D:\\time2image\\%s\\SMTF\\clutter\\', dataName); % 海杂波的保存路径

% STFT 路径
output_dir_STFT_target = sprintf('D:\\time2image\\%s\\STFT\\target\\', dataName); % 目标波的保存路径
output_dir_STFT_clutter = sprintf('D:\\time2image\\%s\\STFT\\clutter\\', dataName); % 海杂波的保存路径

%融合RGB路径

output_dir_RGB_target =sprintf('D:\\time2image\\%s\\fusionRGB\\target\\', dataName); % 目标波的保存路径
output_dir_RGB_clutter = sprintf('D:\\time2image\\%s\\fusionRGB\\clutter\\', dataName); % 海杂波的保存路径

% 检查并创建文件夹
if ~exist(output_dir_GASF_target, 'dir')
    mkdir(output_dir_GASF_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_GASF_clutter, 'dir')
    mkdir(output_dir_GASF_clutter);  % 如果海杂波文件夹不存在，则创建
end
if ~exist(output_dir_GADF_target, 'dir')
    mkdir(output_dir_GADF_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_GADF_clutter, 'dir')
    mkdir(output_dir_GADF_clutter);  % 如果海杂波文件夹不存在，则创建
end

% 检查并创建文件夹
if ~exist(output_dir_SMTF_target, 'dir')
    mkdir(output_dir_SMTF_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_SMTF_clutter, 'dir')
    mkdir(output_dir_SMTF_clutter);  % 如果海杂波文件夹不存在，则创建
end

% 检查并创建文件夹
if ~exist(output_dir_STFT_target, 'dir')
    mkdir(output_dir_STFT_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_STFT_clutter, 'dir')
    mkdir(output_dir_STFT_clutter);  % 如果海杂波文件夹不存在，则创建
end
% 检查并创建文件夹
if ~exist(output_dir_RGB_target, 'dir')
    mkdir(output_dir_RGB_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_RGB_clutter, 'dir')
    mkdir(output_dir_RGB_clutter);  % 如果海杂波文件夹不存在，则创建
end
%%
% 得到并保存GAF图



% 假设 signal_num 是信号的数量，target_select 和 clutter_select 是选择的距离门索引
for i = 1:signal_num
    for j = 1:length(target_select)
        % 保存目标波的GASF和GADF图像
        save_path_GASF = sprintf('%sGASF_%d_%d.png', output_dir_GASF_target, i, target_select(j)); % 图片命名：GASF_信号编号_距离门编号
        save_path_GADF = sprintf('%sGADF_%d_%d.png', output_dir_GADF_target, i, target_select(j)); % 图片命名：GADF_信号编号_距离门编号
        generateGAF(target_seg(:, i, j), save_path_GASF, save_path_GADF);
    end
end
disp("忠橙！");
for i = 1:signal_num
    for j = 1:length(clutter_select)
        % 保存海杂波的GASF和GADF图像
        save_path_GASF = sprintf('%sGASF_%d_%d.png', output_dir_GASF_clutter, i, clutter_select(j)); % 图片命名：GASF_信号编号_距离门编号
        save_path_GADF = sprintf('%sGADF_%d_%d.png', output_dir_GADF_clutter, i, clutter_select(j)); % 图片命名：GADF_信号编号_距离门编号
        generateGAF(clutter_seg(:, i, j), save_path_GASF, save_path_GADF);
    end
end

disp("忠橙！GAF图片已全部保存！");
%%
% 得到并保存SMTF图


% 假设 signal_num 是信号的数量，target_select 和 clutter_select 是选择的距离门索引
for i = 1:signal_num
    for j = 1:length(target_select)
        % 保存目标波的SMTF图像
        save_path_SMTF = sprintf('%sSMTF_%d_%d.png', output_dir_SMTF_target, i, target_select(j)); % 图片命名：GASF_信号编号_距离门编号
        generateSMTF(target_seg(:, i, j), 4,save_path_SMTF);
    end
end
disp("忠橙！");
for i = 1:signal_num
    for j = 1:length(clutter_select)
        % 保存海杂波的SMTF图像
        save_path_SMTF= sprintf('%sSMTF_%d_%d.png', output_dir_SMTF_clutter, i, clutter_select(j)); % 图片命名：GASF_信号编号_距离门编号
        generateSMTF(clutter_seg(:, i, j), 4,save_path_SMTF);
    end
end

disp("忠橙！MTF图片已全部保存！");

%%
%STFT
% 得到并保存STFT图,似乎没有用复数而是用的实数

% 假设 signal_num 是信号的数量，target_select 和 clutter_select 是选择的距离门索引
for i = 1:signal_num
    for j = 1:length(target_select)
        % 保存目标波的STFT图像
        save_path_STFT = sprintf('%sSTFT_%d_%d.png', output_dir_STFT_target, i, target_select(j)); % 图片命名：STFT_信号编号_距离门编号
        generateSTFT(target_seg_raw(:, i, j),save_path_STFT);
    end
end
disp("忠橙！");
for i = 1:signal_num
    for j = 1:length(clutter_select)
        % 保存海杂波的STFT图像
        save_path_STFT= sprintf('%sSTFT_%d_%d.png', output_dir_STFT_clutter, i, clutter_select(j)); % 图片命名：STFT_信号编号_距离门编号
        generateSTFT(clutter_seg_raw(:, i, j),save_path_STFT);
    end
end

disp("忠橙！STFT图片已全部保存！");

%%
%三个RGB图生成RGB图
%在for循环里面得到各种RGB的 地址，然后放到fusion函数，并保存
% 设置加权系数（可以调整为合适的值）
weight = [0.5, 0.3, 0.2];  % STFT、GAF 和 SMTF 的加权




% 对目标波（target_select）的图像进行融合
for i = 1:signal_num
    for j = 1:length(target_select)
        % 读取之前保存的 STFT, GAF 和 SMTF 图像
        stft_path = sprintf('%sSTFT_%d_%d.png', output_dir_STFT_target, i, target_select(j));
        gaf_path = sprintf('%sGASF_%d_%d.png', output_dir_GASF_target, i, target_select(j));
        smtf_path = sprintf('%sSMTF_%d_%d.png', output_dir_SMTF_target, i, target_select(j));

        % 融合图像并保存
        fused_image_path = sprintf('%sFusion_RGB_%d_%d.png', output_dir_RGB_target, i, target_select(j));
        FusionRGB(stft_path, gaf_path, smtf_path, fused_image_path, weight);
    end
end

disp("目标波的融合图像已全部保存！");

% 对海杂波（clutter_select）的图像进行融合
for i = 1:signal_num
    for j = 1:length(clutter_select)
        % 读取之前保存的 STFT, GAF 和 SMTF 图像
        stft_path = sprintf('%sSTFT_%d_%d.png', output_dir_STFT_clutter, i, clutter_select(j));
        gaf_path = sprintf('%sGASF_%d_%d.png', output_dir_GASF_clutter, i, clutter_select(j));
        smtf_path = sprintf('%sSMTF_%d_%d.png', output_dir_SMTF_clutter, i, clutter_select(j));

        % 融合图像并保存
        fused_image_path = sprintf('%sFusion_RGB_%d_%d.png', output_dir_RGB_clutter, i, clutter_select(j));
        FusionRGB(stft_path, gaf_path, smtf_path, fused_image_path, weight);
    end
end

disp("海杂波的融合图像已全部保存！");


