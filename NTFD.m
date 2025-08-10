%%使用各种算法得到NTFD图
close all;
clc
%%
% %%% 绘制 B 显图 
% % 计算信号幅度
% radar_data=complexSeq30HH; 
% signal_magnitude =abs(radar_data); % 取绝对值
% 
% % 将幅度转换为 dB 值（可选） 
% signal_magnitude_db = 20*log10(signal_magnitude); % 转换为dB
% figure; 
% imagesc(1:14, 1:131072, signal_magnitude_db); % 使用 dB 值绘制 
% %imagesc(1:950, 1:131072, signal_magnitude); % 使用幅度绘制 xlabel('RangeGate'); 
% ylabel('Pulse Number'); 
% title('30-B-Scope Display'); 
% colorbar;
% colormap('jet'); % 设置颜色映射 axis xy; % 确保纵轴方向正确

%%
%输入参数
dataName='data280HH';
clutter_select = [1:6,11:14];%选择要处理的距离门
target_select = [8];%根据三特征经典论文，只使用primary cell
win_length = 1024;     % 窗口长度
step_size = 64;        % 步长
signal=abs(complexSeq280HH);%读取实数信号
signal_raw=(complexSeq280HH);%读取复数信号
%%%%%%
all_wave_seg=slidingWindow(signal_raw,1:14,win_length,step_size);%对所有复数信号滑动窗口切片1024*X*14
all_wave_seg_abs=slidingWindow(signal,1:14,win_length,step_size);%对所有复数信号的模值滑动窗口切片1024*X*14
target_seg_raw=slidingWindow(signal_raw,target_select,win_length,step_size);%复数滑动窗口切片
clutter_seg_raw=slidingWindow(signal_raw,clutter_select,win_length,step_size);%复数滑动窗口切片
%%
% SPWVD路径
output_dir_SPWVD_target = sprintf('D:\\time2image\\%s\\NTFD\\SPWVD\\target\\', dataName); % 目标波的保存路径
output_dir_SPWVD_clutter = sprintf('D:\\time2image\\%s\\NTFD\\SPWVD\\clutter\\', dataName); % 海杂波的保存路径
% RP路径
output_dir_RP_target = sprintf('D:\\time2image\\%s\\NTFD\\RP\\target\\', dataName); % 目标波的保存路径
output_dir_RP_clutter = sprintf('D:\\time2image\\%s\\NTFD\\RP\\clutter\\', dataName); % 海杂波的保存路径
%CWT路径
output_dir_CWT_target = sprintf('D:\\time2image\\%s\\NTFD\\CWT\\target\\', dataName); % 目标波的保存路径
output_dir_CWT_clutter = sprintf('D:\\time2image\\%s\\NTFD\\CWT\\clutter\\', dataName); % 海杂波的保存路径
%融合RGB路径
output_dir_RGB_target =sprintf('D:\\time2image\\%s\\NTFD\\fusionRGB\\target\\', dataName); % 目标波的保存路径
output_dir_RGB_clutter = sprintf('D:\\time2image\\%s\\NTFD\\fusionRGB\\clutter\\', dataName); % 海杂波的保存路径
% %融合[a,b,c]RGB路径
% output_dir_RGB_target =sprintf('D:\\time2image\\%s\\NTFD\\221fusionRGB\\target\\', dataName); % 目标波的保存路径
% output_dir_RGB_clutter = sprintf('D:\\time2image\\%s\\NTFD\\221fusionRGB\\clutter\\', dataName); % 海杂波的保存路径
%%
% 检查并创建文件夹
if ~exist(output_dir_SPWVD_target, 'dir')
    mkdir(output_dir_SPWVD_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_SPWVD_clutter, 'dir')
    mkdir(output_dir_SPWVD_clutter);  % 如果海杂波文件夹不存在，则创建
end

% 检查并创建文件夹
if ~exist(output_dir_RP_target, 'dir')
    mkdir(output_dir_RP_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_RP_clutter, 'dir')
    mkdir(output_dir_RP_clutter);  % 如果海杂波文件夹不存在，则创建
end

% 检查并创建文件夹
if ~exist(output_dir_CWT_target, 'dir')
    mkdir(output_dir_CWT_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_CWT_clutter, 'dir')
    mkdir(output_dir_CWT_clutter);  % 如果海杂波文件夹不存在，则创建
end

% 检查并创建文件夹
if ~exist(output_dir_RGB_target, 'dir')
    mkdir(output_dir_RGB_target);  % 如果目标波文件夹不存在，则创建
end
if ~exist(output_dir_RGB_clutter, 'dir')
    mkdir(output_dir_RGB_clutter);  % 如果海杂波文件夹不存在，则创建
end
%% 1.使用SPWVD并根据杂波门归一化得到NTFD图

%对于每个1024长度的信号，使用10个杂波门的SPWVD的均值与方差，对待测信号的SPWVD做归一化
tic;
%保存目标波的SPWVD（NTFD）图像
chooseGate=target_select;  %要检测的距离门
for i=1:size(all_wave_seg,2)%生成图片数量
        save_path_SPWVD = sprintf('%sSPWVD_%d_%d.png', output_dir_SPWVD_target, i, chooseGate); % 图片命名：SPWVD_信号编号_距离门编号
        generateSPWVD(all_wave_seg(:,i,:), chooseGate,clutter_select,save_path_SPWVD)
end

%保存海杂波的SPWVD（NTFD）图像
chooseGate=clutter_select;  %要检测的距离门
for i=1:size(all_wave_seg,2)%生成图片数量
    for j=chooseGate
        save_path_SPWVD = sprintf('%sSPWVD_%d_%d.png', output_dir_SPWVD_clutter, i, j); % 图片命名：SPWVD_信号编号_距离门编号
        generateSPWVD(all_wave_seg(:,i,:), j,clutter_select,save_path_SPWVD)
    end
end
elapsed_time=toc;
fprintf("SPWVD运行时间：%.2f 秒\n",elapsed_time);

%% 2.使用递归图(没有归一化)
tic;
% 假设 signal_num 是信号的数量，target_select 和 clutter_select 是选择的距离门索引
for i = 1:size(all_wave_seg,2)
    for j = 1:length(target_select)
        % 保存目标波的RP图像
        save_path_RP = sprintf('%sRP_%d_%d.png', output_dir_RP_target, i, target_select(j)); % 图片命名：RP_信号编号_距离门编号
        generateRP(target_seg_raw(:, i, j), save_path_RP);
    end
end

for i = 1:size(all_wave_seg,2)
    for j = 1:length(clutter_select)
        % 保存海杂波的RP图像
        save_path_RP = sprintf('%sRP_%d_%d.png', output_dir_RP_clutter, i, clutter_select(j)); % 图片命名：RP_信号编号_距离门编号
        generateRP(clutter_seg_raw(:, i, j), save_path_RP);
    end
end
elapsed_time=toc;
fprintf("RP运行时间：%.2f 秒\n",elapsed_time);
% tic;
% %保存目标波的CWT（NTFD）图像
% chooseGate=target_select;  %要检测的距离门(目标)
% for i=1:size(all_wave_seg_abs,2)%生成图片数量
%         save_path_RP = sprintf('%sRP_%d_%d.png', output_dir_RP_target, i, chooseGate); % 图片命名：RP_信号编号_距离门编号
%         generateNRP(all_wave_seg_abs(:,i,:), chooseGate,clutter_select,save_path_RP)
% end
% 
% %保存海杂波的SPWVD（NTFD）图像
% chooseGate=clutter_select;  %要检测的距离门（杂波）
% for i=1:size(all_wave_seg_abs,2)%生成图片数量
%     for j=chooseGate
%         save_path_RP = sprintf('%sRP_%d_%d.png', output_dir_RP_clutter, i, j); % 图片命名：RP_信号编号_距离门编号
%         generateNRP(all_wave_seg_abs(:,i,:), j,clutter_select,save_path_RP)
%     end
% end
% elapsed_time=toc;
% fprintf("CWT运行时间：%.2f 秒\n",elapsed_time);



%% 3.使用CWT并根据杂波门归一化得到NTFD图

%对于每个1024长度的信号，使用10个杂波门的CWT的均值与方差，对待测信号的CWT做归一化
% 得到并保存CWT图
tic;
%保存目标波的CWT（NTFD）图像
chooseGate=target_select;  %要检测的距离门(目标)
for i=1:size(all_wave_seg_abs,2)%生成图片数量
        save_path_RP = sprintf('%sCWT_%d_%d.png', output_dir_CWT_target, i, chooseGate); % 图片命名：SPWVD_信号编号_距离门编号
        generateCWT(all_wave_seg_abs(:,i,:), chooseGate,clutter_select,save_path_RP)
end

%保存海杂波的SPWVD（NTFD）图像
chooseGate=clutter_select;  %要检测的距离门（杂波）
for i=1:size(all_wave_seg_abs,2)%生成图片数量
    for j=chooseGate
        save_path_RP= sprintf('%sCWT_%d_%d.png', output_dir_CWT_clutter, i, j); % 图片命名：SPWVD_信号编号_距离门编号
        generateCWT(all_wave_seg_abs(:,i,:), j,clutter_select,save_path_RP)
    end
end
elapsed_time=toc;
fprintf("CWT运行时间：%.2f 秒\n",elapsed_time);

%% 4.图像融合
%读取三种方法产生的灰度图，并加权完成图像融合
tic;
%三个灰度图生成RGB图
%在for循环里面得到各种RGB的 地址，然后放到fusion函数，并保存
% 设置加权系数（可以调整为合适的值）
weight = [1,1,1];  % SPWVD、RP 和 CWT 的加权




% 对目标波（target_select）的图像进行融合
for i = 1:size(all_wave_seg_abs,2)
    for j = 1:length(target_select)
        % 读取之前保存的SPWVD,RP,CWT 图像
        SPWVD_path = sprintf('%sSPWVD_%d_%d.png', output_dir_SPWVD_target, i, target_select(j));
        RP_path = sprintf('%sRP_%d_%d.png', output_dir_RP_target, i, target_select(j));
        CWT_path = sprintf('%sCWT_%d_%d.png', output_dir_CWT_target, i, target_select(j));

        % 融合图像并保存
        fused_image_path = sprintf('%sFusion_RGB%d_%d.png', output_dir_RGB_target, i, target_select(j));
        FusionRGB(SPWVD_path, RP_path, CWT_path, fused_image_path, weight);
    end
end

disp("目标波的融合图像已全部保存！");

% 对海杂波（clutter_select）的图像进行融合
for i = 1:size(all_wave_seg_abs,2)
    for j = 1:length(clutter_select)
        % 读取之前保存的 STFT, GAF 和 SMTF 图像
        SPWVD_path = sprintf('%sSPWVD_%d_%d.png', output_dir_SPWVD_clutter, i, clutter_select(j));
        RP_path = sprintf('%sRP_%d_%d.png', output_dir_RP_clutter, i, clutter_select(j));
        CWT_path = sprintf('%sCWT_%d_%d.png', output_dir_CWT_clutter, i, clutter_select(j));

        % 融合图像并保存
        fused_image_path = sprintf('%sFusion_RGB_%d_%d.png', output_dir_RGB_clutter, i, clutter_select(j));
        FusionRGB(SPWVD_path, RP_path, CWT_path, fused_image_path, weight);
    end
end

disp("海杂波的融合图像已全部保存！");
elapsed_time=toc;
fprintf("图像融合运行时间：%.2f 秒\n",elapsed_time);
! shutdown -s